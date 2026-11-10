#!/usr/bin/env python3
"""Gate-2 waist-angle pipeline -- command-line entry point.

  live ROS 2 (drone or `ros2 bag play`):
      python3 run_gate2.py live [--use-sim-time]
  rosbag2 read directly (every frame, deterministic):
      python3 run_gate2.py bag ~/rosbags/25m_2x_1.5ms --start 60 --duration 30
  plain video / camera, no telemetry:
      python3 run_gate2.py video clip.mp4

Common options: --config FILE, --set section.key=value (repeatable),
--no-display, --save-video [PATH], --save-frames DIR, --no-lidar (any mode:
no lidar at all, relative Depth-Anything depth, attitude-normalised frame). Frames with detected
people are saved (annotated JPEG + JSON) to <output.dir>/<run>_frames/ by
default. Keys: space/p pause-resume (all modes, live included), n step one
frame while paused, s snapshot, q/Esc quit. See README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gate2.config import load_config  # noqa: E402
from gate2.utils import log  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["live", "bag", "video"])
    ap.add_argument("path", nargs="?", help="bag directory (bag) or file/camera index (video)")
    ap.add_argument("--config", help="YAML overriding config/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one config key, e.g. --set depth.tile_res=504")
    ap.add_argument("--use-sim-time", action="store_true", help="live: follow /clock (bag replay)")
    ap.add_argument("--start", type=float, default=0.0, help="bag/video: start offset, s")
    ap.add_argument("--duration", type=float, default=None, help="bag: seconds to process")
    ap.add_argument("--stride", type=int, default=1, help="bag/video: process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--realtime", action="store_true",
                    help="bag/video: pace to recorded time and drop frames like live")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-video", nargs="?", const="auto", default=None, metavar="PATH",
                    help="write the annotated video (default <output.dir>/<run>.mp4)")
    ap.add_argument("--no-log", action="store_true", help="do not write the JSONL log")
    ap.add_argument("--save-frames", metavar="DIR",
                    help="directory for frames with detections (default <output.dir>/<run>_frames)")
    ap.add_argument("--no-save-frames", action="store_true", help="do not save detected frames")
    ap.add_argument("--save-raw-frames", action="store_true",
                    help="also save the unannotated frame next to each annotated one")
    ap.add_argument("--dump", metavar="DIR",
                    help="save each person's keypoints, DA3 relief, crop and telemetry (npz)")
    ap.add_argument("--zoom", choices=["1x", "2x"], help="shortcut for geo.intrinsics fcb_1x/fcb_2x")
    ap.add_argument("--pose-model", choices=["h", "s"], help="shortcut for pose.model")
    ap.add_argument("--no-depth", action="store_true", help="disable DA3 (fallback methods only)")
    ap.add_argument("--no-lidar", action="store_true",
                    help="no lidar at all; relative DA3 depth in an attitude-normalised frame "
                         "(gate2/no_lidar.py, config/no_lidar.yaml)")
    return ap.parse_args()


def show_frame(cv2, cfg, img, paused):
    """Show one frame; when paused, draw a banner and wait for a key. Returns the key."""
    w = cfg.display.width
    disp = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1]))) if img.shape[1] != w else img
    if paused:
        disp = disp.copy()
        msg = "PAUSED  space/p resume   n step   s snapshot   q quit"
        cv2.rectangle(disp, (0, disp.shape[0] - 34), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
        cv2.putText(disp, msg, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 220, 255), 2, cv2.LINE_AA)
    cv2.imshow(cfg.display.window_name, disp)
    # paused: poll so the window stays responsive (and a live feed keeps draining)
    return cv2.waitKey(50 if paused else 1) & 0xFF


def main():
    a = parse()
    sets = list(a.set)
    if a.zoom:
        sets.append(f"geo.intrinsics=fcb_{a.zoom}")
    if a.pose_model:
        sets.append(f"pose.model={a.pose_model}")
    if a.no_depth:
        sets.append("depth.enabled=false")
    if a.save_video:
        sets.append(f"output.video={a.save_video}")
    if a.no_log:
        sets.append("output.log=null")
    if a.save_frames:
        sets.append(f"output.frames={a.save_frames}")
    if a.no_save_frames:
        sets.append("output.frames=null")
    if a.save_raw_frames:
        sets.append("output.frames_raw=true")
    if a.no_display:
        sets.append("display.enabled=false")
    layers = [os.path.join(HERE, "config", "no_lidar.yaml")] if a.no_lidar else []
    cfg = load_config(a.config, sets, layers)
    if a.mode == "live" and not any(x.startswith("depth.tile_res=") for x in sets):
        cfg.depth.tile_res = cfg.depth.tile_res_live

    if a.mode in ("bag", "video") and not a.path:
        sys.exit(f"{a.mode} mode needs a path")
    if a.mode == "bag":
        from gate2.sources.bag_offline import BagSource
        src = BagSource(cfg, a.path, a.start, a.duration, a.stride, a.max_frames, a.realtime)
        run = f"{src.name}_{time.strftime('%Y%m%d-%H%M%S')}{'_nolidar' if a.no_lidar else ''}"
    elif a.mode == "video":
        from gate2.sources.video import VideoSource
        src = VideoSource(cfg, a.path, a.start, a.stride, a.max_frames, a.realtime)
        run = f"{os.path.splitext(os.path.basename(str(a.path)))[0]}_{time.strftime('%Y%m%d-%H%M%S')}{'_nolidar' if a.no_lidar else ''}"
    else:
        from gate2.sources.ros_live import RosLiveSource
        src = RosLiveSource(cfg, use_sim_time=a.use_sim_time)
        run = f"live_{time.strftime('%Y%m%d-%H%M%S')}{'_nolidar' if a.no_lidar else ''}"

    from gate2.pipeline import Gate2Pipeline, Outputs
    pipe = Gate2Pipeline(cfg)
    outs = Outputs(cfg, run)

    show = cfg.display.enabled and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if cfg.display.enabled and not show:
        log("warn", "display", "no $DISPLAY: running headless (use --save-video to see results)")
    import cv2
    if show:
        try:
            cv2.namedWindow(cfg.display.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as e:   # e.g. opencv-python-headless
            log("warn", "display", f"cannot open a window ({e}); running headless")
            show = False
    n, paused, step = 0, False, False
    t0 = time.monotonic()
    try:
        for frame, stamp in src.frames():
            tel = src.hub.snapshot(stamp)
            fr = pipe.process(frame, stamp, tel, src.hub.camera_K)
            age = (src.now() - stamp) if hasattr(src, "now") else None
            img = pipe.render(fr, f"[{a.mode}{' no-lidar' if a.no_lidar else ''}] {src.name}", age_s=age)
            recs = pipe.record(fr)
            if a.dump:
                pipe.dump(fr, a.dump, n)
            outs.write(img, recs, fps=max(pipe.rate.hz, 1.0) if src.live else 30.0 / a.stride)
            if hasattr(src, "publish"):
                src.publish(img, stamp, json.dumps({"stamp": stamp, "people": recs}))
            n += 1
            if n % 20 == 0 or n == 1:
                ang = [f"#{r['track']}:{r['hip_angle_smoothed_deg']}({r['method']})" for r in recs]
                log("info", "run", f"frame {n}  {pipe.rate.hz:.2f} Hz  people {len(recs)}  "
                                   f"{' '.join(ang)}  {src.stats()}")
            saved = outs.save_frame(n, stamp, img, frame, recs)
            if show:
                key = show_frame(cv2, cfg, img, paused or step)
                step = False
                while True:
                    if key in (ord("q"), 27):
                        raise StopIteration
                    if key == ord("s"):
                        p = os.path.join(cfg.output.dir, f"{run}_snap{n:05d}.png")
                        cv2.imwrite(p, img)
                        log("info", "run", f"snapshot {p}" + (f" (also in {saved})" if saved else ""))
                    if key in (ord(" "), ord("p")):
                        paused = not paused
                        log("info", "run", "paused" if paused else "resumed")
                    if key in (ord("n"), ord(".")) and paused:
                        step = True          # process exactly one more frame, then pause
                        break
                    if not paused:
                        break
                    key = show_frame(cv2, cfg, img, True)
            if a.max_frames and n >= a.max_frames:
                break
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        outs.close()
        src.close()
        if show:
            cv2.destroyAllWindows()
        log("info", "run", f"{n} frames in {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
