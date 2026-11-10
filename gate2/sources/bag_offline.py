"""Read a rosbag2 directly: every frame, deterministic, faster or slower than real time.

This is the path for analysing a recording. Nothing is played over DDS; the
bag is read in recorded order, telemetry messages go into the TelemetryHub
keyed on their header stamps, and image packets are decoded in-process.

A frame is released only once the bag clock has run `lookahead_s` past its
stamp, so the IMU samples *after* the shutter are already buffered and the
attitude can be interpolated rather than extrapolated (on the fcb bags the
image is logged ~50 ms after its stamp, the IMU within ~1 ms of its own).

Everything comes from bag topics -- no sidecar CSV or mp4 is read.
"""
from __future__ import annotations

import os
import time
from collections import deque

import yaml

from ..image_codec import make_image_decoder, stamp_s
from ..telemetry import TelemetryHub
from ..utils import log


def _metadata(bag):
    """(storage_id, start_time_ns or None) from metadata.yaml."""
    try:
        with open(os.path.join(bag, "metadata.yaml")) as f:
            info = yaml.safe_load(f)["rosbag2_bagfile_information"]
        return (info["storage_identifier"],
                int(info.get("starting_time", {}).get("nanoseconds_since_epoch", 0)) or None)
    except Exception:  # noqa: BLE001
        sid = "mcap" if any(p.endswith(".mcap") for p in os.listdir(bag)) else "sqlite3"
        return sid, None


class BagSource:
    live = False

    def __init__(self, cfg, bag, start_s=0.0, duration_s=None, stride=1, max_frames=None,
                 realtime=False, lookahead_s=0.15):
        import rosbag2_py
        from rosidl_runtime_py.utilities import get_message
        self.cfg, self.bag = cfg, os.path.expanduser(bag)
        self.name = os.path.basename(os.path.normpath(self.bag))
        self.hub = TelemetryHub(cfg, live=False)
        self.start_s, self.duration_s = start_s, duration_s
        self.stride, self.max_frames, self.realtime = max(1, stride), max_frames, realtime
        self.lookahead = lookahead_s
        sid, self.t0_ns = _metadata(self.bag)
        self.reader = rosbag2_py.SequentialReader()
        try:
            self.reader.open(rosbag2_py.StorageOptions(uri=self.bag, storage_id=sid),
                             rosbag2_py.ConverterOptions("", ""))
        except Exception as e:
            hint = (" -- install the plugin: sudo apt install ros-humble-rosbag2-storage-mcap"
                    if sid == "mcap" else "")
            raise RuntimeError(f"cannot open bag {self.bag} ({sid}): {e}{hint}") from e
        types = {t.name: t.type for t in self.reader.get_all_topics_and_types()}
        r = cfg.ros

        # image topic: the configured one, else the first fallback present
        cands = [r.image_topic] + list(r.image_fallback_topics)
        img = next((t for t in cands if t in types), None)
        if img is None:
            img = next((t for t, ty in types.items()
                        if ty.endswith(("FFMPEGPacket", "CompressedImage", "/Image"))), None)
        if img is None:
            raise RuntimeError(f"no image topic in {self.bag}; topics: {sorted(types)}")
        if img != r.image_topic:
            log("warn", "source", f"{r.image_topic} not in bag, using {img} ({types[img]})")
        self.image_topic = img
        self.decode, self.transport = make_image_decoder(types[img])
        self.handlers = {img: None}
        want = {r.imu_topic: "imu", r.rel_alt_topic: "rel_alt", r.gps_topic: "gps",
                r.compass_topic: "compass", r.scan_topic: "scan", r.pose_topic: "pose",
                r.camera_info_topic: "info"}
        present = []
        for t, kind in want.items():
            if not t:
                continue          # disabled in the config (e.g. scan in --no-lidar)
            if t in types:
                self.handlers[t] = kind
                present.append(kind)
            else:
                log("warn", "source", f"bag has no {t} ({kind}) -- its fallback will be used")
        self.msg_types = {t: get_message(types[t]) for t in self.handlers}
        self.reader.set_filter(rosbag2_py.StorageFilter(topics=list(self.handlers)))
        # Seek near the start instead of decoding everything before it: 2 s of
        # pre-roll fills the telemetry buffers, and the decoder waits for the
        # next keyframe (every 0.5 s on the FCB stream).
        self.preroll = 2.0
        if start_s > self.preroll and self.t0_ns and hasattr(self.reader, "seek"):
            try:
                self.reader.seek(self.t0_ns + int((start_s - self.preroll) * 1e9))
            except Exception as e:  # noqa: BLE001
                log("warn", "source", f"seek failed ({e}); reading from the start")
        log("info", "source", f"bag {self.name}: image {img} [{self.transport}], "
                              f"telemetry {', '.join(present) or 'none'}")

    def _feed(self, kind, m, t_bag):
        h = self.hub
        if kind == "imu":
            q = m.orientation
            h.on_imu(stamp_s(m.header), q.x, q.y, q.z, q.w)
        elif kind == "rel_alt":
            h.on_rel_alt(t_bag, m.data)
        elif kind == "compass":
            h.on_compass(t_bag, m.data)
        elif kind == "gps":
            h.on_gps(stamp_s(m.header), m.latitude, m.longitude, m.status.status >= 0)
        elif kind == "scan":
            import numpy as np
            ang = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
            h.on_scan(stamp_s(m.header), ang, m.ranges, m.range_min, m.range_max)
        elif kind == "pose":
            h.on_pose(stamp_s(m.header), m.pose.position.z)
        elif kind == "info":
            h.on_camera_info(list(m.k), m.width, m.height)

    def frames(self):
        from rclpy.serialization import deserialize_message
        pending = deque()
        t_first = None
        n_img = n_out = 0
        wall0 = bag0 = None
        while self.reader.has_next():
            topic, data, t_ns = self.reader.read_next()
            t_bag = t_ns * 1e-9
            if t_first is None:
                t_first = self.t0_ns * 1e-9 if self.t0_ns else t_bag
            rel = t_bag - t_first
            if self.duration_s and rel > self.start_s + self.duration_s + self.lookahead:
                break
            kind = self.handlers[topic]
            try:
                m = deserialize_message(data, self.msg_types[topic])
                if kind is None:
                    # every packet is decoded (inter-coded stream), frames are skipped later
                    for frame, stamp in self.decode(m):
                        n_img += 1
                        if rel >= self.start_s and (n_img - 1) % self.stride == 0:
                            pending.append((frame, stamp))
                else:
                    self._feed(kind, m, t_bag)
            except Exception as e:  # noqa: BLE001 - one bad message must not end the run
                log("warn", "source", f"{topic}: {type(e).__name__}: {e}")
            while pending and pending[0][1] + self.lookahead <= t_bag:
                frame, stamp = pending.popleft()
                if self.realtime:
                    if wall0 is None:
                        wall0, bag0 = time.monotonic(), stamp
                    ahead = (stamp - bag0) - (time.monotonic() - wall0)
                    if ahead > 0:
                        time.sleep(ahead)
                    elif ahead < -0.1 and pending:
                        continue   # behind real time: drop, as live would
                yield frame, stamp
                n_out += 1
                if self.max_frames and n_out >= self.max_frames:
                    return
        while pending:
            yield pending.popleft()

    def stats(self):
        c = self.hub.counts
        return "bag rx " + " ".join(f"{k}={v}" for k, v in c.items() if v)

    def close(self):
        pass
