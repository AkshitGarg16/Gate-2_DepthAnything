"""Live ROS 2 source -- the drone, or a bag replayed with `ros2 bag play --clock`.

Subscribes to the camera and MAVROS/lidar topics, buffers telemetry on header
stamps, and hands the newest frame to the pipeline. It never queues frames:
if inference is slower than the camera, intermediate frames are dropped and
the HUD shows the age of what was processed. A queue would drift ever further
behind real time while reporting a healthy frame rate.

Image transport (ros.image_transport: auto) is chosen from the type actually
advertised on the topic: sensor_msgs/Image, CompressedImage, or the FCB's
ffmpeg_image_transport_msgs/FFMPEGPacket, which is decoded here with PyAV
(no `image_transport republish` process needed). If ros.image_topic is not
advertised, ros.image_fallback_topics are tried in order.

QoS: the image subscription matches the publishers' reliability (a RELIABLE
subscriber never matches a BEST_EFFORT publisher; a 6 MB raw frame over
BEST_EFFORT is lost on any dropped fragment -- Gate-1 measured 5.7 Hz vs
17-20 Hz). Telemetry subscribes BEST_EFFORT, which matches either.

Callbacks never raise: an exception on the executor thread kills the spin
thread silently and the node looks hung. Errors are counted and reported by
the periodic health line instead.
"""
from __future__ import annotations

import threading
import time

from ..image_codec import make_image_decoder, stamp_s
from ..telemetry import TelemetryHub
from ..utils import log


class RosLiveSource:
    live = True

    def __init__(self, cfg, use_sim_time=False, wait_s=10.0):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        self.cfg = cfg
        self.hub = TelemetryHub(cfg, live=True)
        self.name = "ros-live"
        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.node = Node(cfg.ros.node_name)
        if use_sim_time:
            from rclpy.parameter import Parameter
            self.node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.QoS, self.Rel, self.Hist = QoSProfile, ReliabilityPolicy, HistoryPolicy
        self.sensor_qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                                     history=HistoryPolicy.KEEP_LAST)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.latest = None
        self.st = {"rx": 0, "frames": 0, "drop": 0, "err": 0, "errmsg": "", "served": 0}
        self._subscribe_telemetry()
        self._stop = False
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()
        self._subscribe_image(wait_s)
        self.pub_img = self.pub_res = None
        self._make_publishers()

    # -- subscriptions -------------------------------------------------------
    def _guard(self, fn):
        def cb(m):
            try:
                fn(m)
            except Exception as e:  # noqa: BLE001
                self.st["err"] += 1
                self.st["errmsg"] = f"{type(e).__name__}: {e}"
        return cb

    def _subscribe_telemetry(self):
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Imu, LaserScan, NavSatFix
        from std_msgs.msg import Float64
        import numpy as np
        r, h, n = self.cfg.ros, self.hub, self.node
        now = lambda: n.get_clock().now().nanoseconds * 1e-9  # noqa: E731

        def imu(m):
            q = m.orientation
            h.on_imu(stamp_s(m.header), q.x, q.y, q.z, q.w)

        def scan(m):
            ang = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
            h.on_scan(stamp_s(m.header), ang, m.ranges, m.range_min, m.range_max)

        subs = [
            (Imu, r.imu_topic, imu),
            (Float64, r.rel_alt_topic, lambda m: h.on_rel_alt(now(), m.data)),
            (Float64, r.compass_topic, lambda m: h.on_compass(now(), m.data)),
            (NavSatFix, r.gps_topic,
             lambda m: h.on_gps(stamp_s(m.header), m.latitude, m.longitude, m.status.status >= 0)),
            (LaserScan, r.scan_topic, scan),
            (PoseStamped, r.pose_topic, lambda m: h.on_pose(stamp_s(m.header), m.pose.position.z)),
            (CameraInfo, r.camera_info_topic, lambda m: h.on_camera_info(list(m.k), m.width, m.height)),
        ]
        for typ, topic, fn in subs:
            if topic:
                n.create_subscription(typ, topic, self._guard(fn), self.sensor_qos)

    def _topic_types(self):
        return {t: ty[0] for t, ty in self.node.get_topic_names_and_types() if ty}

    def _subscribe_image(self, wait_s):
        from rosidl_runtime_py.utilities import get_message
        r = self.cfg.ros
        cands = [r.image_topic] + [t for t in r.image_fallback_topics if t != r.image_topic]
        topic = typ = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s:
            types = self._topic_types()
            for t in cands:
                if t in types:
                    topic, typ = t, types[t]
                    break
            if topic:
                break
            time.sleep(0.25)
        forced = {"raw": "sensor_msgs/msg/Image", "compressed": "sensor_msgs/msg/CompressedImage",
                  "ffmpeg": "ffmpeg_image_transport_msgs/msg/FFMPEGPacket"}
        if r.image_transport != "auto":
            typ = forced[r.image_transport]
            topic = topic or r.image_topic
        if topic is None:
            # nothing advertised yet: subscribe to the configured topic and wait
            topic = r.image_topic
            typ = forced["ffmpeg"] if topic.endswith("ffmpeg") else forced["raw"]
            log("warn", "source", f"none of {cands} is advertised after {wait_s:.0f}s; "
                                  f"subscribing to {topic} as {typ} and waiting")
        elif topic != r.image_topic:
            log("warn", "source", f"{r.image_topic} not advertised, using {topic}")
        self.decode, self.transport = make_image_decoder(typ)
        self.image_topic = topic

        rel = self.Rel.BEST_EFFORT
        pubs = self.node.get_publishers_info_by_topic(topic)
        if pubs and all(p.qos_profile.reliability == self.Rel.RELIABLE for p in pubs):
            rel = self.Rel.RELIABLE
        # ffmpeg packets must all arrive (inter-coded); raw frames only need the newest
        depth = 50 if self.transport == "ffmpeg" else 1
        qos = self.QoS(depth=depth, reliability=rel, history=self.Hist.KEEP_LAST)

        def on_img(m):
            self.st["rx"] += 1
            for frame, stamp in self.decode(m):
                with self.cond:
                    if self.latest is not None:
                        self.st["drop"] += 1
                    self.latest = (frame, stamp)
                    self.st["frames"] += 1
                    self.cond.notify()

        self.node.create_subscription(get_message(typ), topic, self._guard(on_img), qos)
        log("info", "source", f"image {topic} [{self.transport}, "
                              f"{'RELIABLE' if rel == self.Rel.RELIABLE else 'BEST_EFFORT'}, "
                              f"{len(pubs)} publisher(s)]")

    def _make_publishers(self):
        r = self.cfg.ros
        if r.publish_image_topic:
            from sensor_msgs.msg import Image
            self.pub_img = self.node.create_publisher(Image, r.publish_image_topic, 1)
        if r.publish_result_topic:
            from std_msgs.msg import String
            self.pub_res = self.node.create_publisher(String, r.publish_result_topic, 10)

    def _spin(self):
        from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
        ex = SingleThreadedExecutor()
        ex.add_node(self.node)
        try:
            while not self._stop and self.rclpy.ok():
                ex.spin_once(timeout_sec=0.1)
        except (ExternalShutdownException, KeyboardInterrupt):
            pass

    # -- consumer side ---------------------------------------------------------
    def frames(self):
        last_report = time.monotonic()
        while not self._stop and self.rclpy.ok():
            with self.cond:
                self.cond.wait_for(lambda: self.latest is not None or self._stop, timeout=0.5)
                item, self.latest = self.latest, None
            if time.monotonic() - last_report > self.cfg.ros.stats_every_s:
                last_report = time.monotonic()
                if item is None or self.st["err"]:
                    log("warn" if item is None else "info", "source", self.health())
            if item is not None:
                self.st["served"] += 1
                yield item

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def health(self):
        s = self.st
        npub = len(self.node.get_publishers_info_by_topic(self.image_topic))
        msg = (f"{self.image_topic}: {npub} publisher(s), {s['rx']} msgs, {s['frames']} frames, "
               f"{s['drop']} dropped (inference slower than camera), {s['err']} callback errors")
        if s["errmsg"]:
            msg += f" (last: {s['errmsg']})"
        if npub == 0:
            msg += " -- nothing is publishing the camera topic"
        if s["rx"] and not s["frames"] and self.transport == "ffmpeg":
            msg += " -- packets arriving but no frame yet (waiting for a keyframe)"
        return msg + " | telemetry " + " ".join(f"{k}={v}" for k, v in self.hub.counts.items())

    def stats(self):
        return f"dropped {self.st['drop']}"

    def publish(self, annotated, stamp_s_, payload_json):
        from builtin_interfaces.msg import Time
        from ..image_codec import bgr_to_imgmsg
        t = Time(sec=int(stamp_s_), nanosec=int((stamp_s_ % 1) * 1e9))
        if self.pub_img is not None:
            self.pub_img.publish(bgr_to_imgmsg(annotated, t))
        if self.pub_res is not None:
            from std_msgs.msg import String
            self.pub_res.publish(String(data=payload_json))

    def close(self):
        self._stop = True
        with self.cond:
            self.cond.notify_all()
        try:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
