# Gate-2 — Waist (hip) angle of people seen from a drone

Gate-2 finds people in a drone's downward-looking camera feed and measures, for
each person, **the angle between the hip→shoulder line and the hip→ankle line**
(hip→knee when the ankle is not detected): how far the
person is bent at the waist. It runs live on ROS 2 topics from the aircraft and
on recorded rosbags. It shows an annotated video feed, logs every measurement,
and publishes the results back on ROS.

It combines three existing pieces:

| Model / pipeline | Where it lives | What Gate-2 uses it for |
|---|---|---|
| **FlyPose** (detector + ViTPose-H "flypose_h") | `~/fly_pose/model` | find people, 17 keypoints per person (2D) |
| **Depth-Anything-3** (DA3 mono-large) | `~/Depth-Anything-3` | relative depth of each person: which joints are raised, and by how much |
| **Gate-1 geolocalisation** (`CoordinateTransformer`) | `~/GATE_1_PIPELINE/deployment_gate_1` | camera rays → metres on the ground, attitude correction, casualty GPS |

```
                       ┌──────────────────────────── Gate-2 ────────────────────────────┐
 camera  ──────────────►  1 FlyPose          boxes + 17 keypoints (pixels)              │
 (Image / Compressed /  │        │                                                       │
  FFMPEG packets)       │        ▼                                                       │
                        │  2 DA3 on a zoomed crop per person → relief map                │
 IMU, rel_alt, GPS,     │        │        (height above a fitted ground plane, DA3 units)│
 lidar, pose, compass ──►  3 Gate-1 rays: keypoint pixel → 3D point at its height (NED) │
                        │        │                                                       │
                        │        ▼                                                       │
                        │  4 3D hip→shoulder & hip→ankle vectors → angle + orientation   │
                        │        │                                                       │
                        │        ▼                                                       │
                        │  5 tracker (ground position) → rolling median per person      │
                        └────────┬───────────────────────────────────────────────────────┘
                                 ▼
                 window / mp4 · JSONL log · /gate2/annotated · /gate2/waist_angles
```

---

## Contents

1. [Quick start](#1-quick-start)
2. [What the number means](#2-what-the-number-means)
3. [How it works, stage by stage](#3-how-it-works-stage-by-stage)
4. [Fallbacks](#4-fallbacks)
5. [Running it](#5-running-it)
6. [Outputs](#6-outputs)
7. [Configuration](#7-configuration)
8. [Performance](#8-performance)
9. [Validation, and what it can and cannot do](#9-validation-and-what-it-can-and-cannot-do)
10. [Troubleshooting](#10-troubleshooting)
11. [Code layout and extending it](#11-code-layout-and-extending-it)

---

## 1. Quick start

```bash
cd ~/Gate-2_Pipeline
source /opt/ros/humble/setup.bash          # Jazzy on the Jetson

python3 tools/check_env.py                 # what each stage will run on, before running

# a recorded bag, read directly (every frame, deterministic)
python3 run_gate2.py bag ~/rosbags/25m_2x_1.5ms --start 85 --duration 30

# a recorded bag replayed over ROS, through the *live* code path
OFFSET=85 ./run_bag.sh ~/rosbags/25m_2x_1.5ms

# on the aircraft
./run_live.sh
```

Keys in the window (all modes, live included): **space / p** pause–resume, **n** step one
frame while paused, **s** snapshot, **q / Esc** quit. Every frame with a detected person is
saved (annotated JPEG + JSON) to `outputs/<run>_frames/`.

---

## 2. What the number means

The reported value is the **hip angle**: the angle at the hip between

* the **torso**: mid-hip → mid-shoulder (the trunk midline), and
* the **leg**: hip → **ankle**, one per side (L and R), combined by keypoint confidence.
  When a side's ankle is not detected, hip → **knee** is used for that side instead
  (logged in `leg_joint` and the notes).

| hip angle | posture |
|---|---|
| ~180° | body straight: lying flat (or standing) |
| ~90°  | torso and leg at a right angle: sitting upright, legs out |
| < 90° | folded: hunched forward over the legs |

The hip→ankle line is not the thigh when the knee is bent. Someone sitting with
the knees hugged up has their ankles on the ground near the buttocks, so their
hip→ankle line is nearly flat and the angle is ~90–110°, although the thigh
itself points up.

The overlay also shows **bend = 180° − hip angle**, i.e. how far the waist is
bent away from straight.

Why the trunk midline and not a per-side shoulder→hip line: the shoulders are
wider than the hips, so a per-side line meets the leg at ~173° even when the
body is perfectly straight. The midline is the anatomical definition of hip
flexion and reads 180° for a straight body.

### Orientation of the two vectors in 3D

For every person, the **hip→shoulder** vector and each **hip→ankle** vector are
also reported as directions in space (local north-east-down frame from the
Gate-1 transformer):

| field | meaning |
|---|---|
| `elevation_deg` | angle above (+) or below (−) the ground plane |
| `from_vertical_deg` | angle from the vertical (90 − \|elevation\|) |
| `azimuth_deg` | compass bearing of the vector's horizontal part (needs heading) |
| `relation` | **parallel** to the ground if \|elevation\| ≤ 20°, **perpendicular** if ≥ 70°, else **oblique** (`angle.parallel_max_deg`, `perpendicular_min_deg`) |
| `direction` | up / down / level |
| `aligned` | true when the two vectors form one straight line (hip angle ≥ `angle.aligned_min_deg`, 165°) |
| `measured` | false for `3d_equal_depth`, where every elevation is 0 *by assumption* |

On screen this is two lines under the angle, e.g.
`hip-shoulder +80 deg perpendicular` / `hip-ankle L -4 parallel  R -2 parallel`
(medians over the track's recent frames).

Typical readings: lying flat → both parallel and aligned; sitting upright with
legs out → torso perpendicular (up), legs parallel; standing (`posture:
upright`) → torso perpendicular up, legs perpendicular down, aligned.

### The angle is always measured between 3D vectors

Every method computes `angle = arccos(T·L / |T||L|)` with **T** and **L**
3D vectors built from 3D keypoints, never from pixel positions. Consequences:

* **Keypoints at equal depth give the correct angle.** For a person lying flat,
  every joint is at the same height above the ground, the vertical components
  are zero, and the result is exactly the angle in the ground plane. It is in
  metres, so it is also correct when the camera is tilted, where the raw pixel
  angle is distorted by perspective (tested to < 1° with 14° of tilt; see §9).
* **Keypoints at different depths are not foreshortened.** A person sitting up
  has a torso pointing at the camera. Its image is nearly zero length and the
  pixel angle is meaningless, but the 3D vector has its true length.

The raw pixel angle is still written to the log (`angle_2d_deg`), for comparison only.

---

## 3. How it works, stage by stage

### Stage 1: FlyPose: people and 2D keypoints (`gate2/pose_estimator.py`)

FlyPose is top-down. An RT-DETR-style detector (fixed 1280×1280 input)
proposes person boxes, then ViTPose (H by default, S as fallback) predicts 17
COCO keypoints per box from a 256×192 crop. Gate-2 imports FlyPose's own
`inference.py` unchanged, so pre- and post-processing match
`run_image_inference.py` exactly. Gate-2-specific choices:

* `letterbox` resize (aspect-preserving) instead of FlyPose's original stretch;
  aerial frames are 16:9 and stretching distorts small people.
* only the `pose.max_persons` most confident boxes go to the pose model, which is
  the slow part on CPU;
* keypoints below `pose.kp_thr` (0.3) count as missing.

### Stage 2: Depth-Anything-3: relief of each person (`gate2/depth_estimator.py`)

For each person a **square crop** around the box is taken, with a ring of
ground around it (`depth.margin`). DA3 runs on that crop alone at
`depth.tile_res` (1008 px offline, 504 px live). Then:

```
plane(u,v)  = robust plane fit of DA3 depth over the ground ring (outside the box)
relief(u,v) = plane(u,v) − depth(u,v)        (+ = above the ground)
```

Why zoom: DA3 samples once per 14 px of its input. A 150 px person in a full
1920×1080 frame gets only a handful of depth samples; the zoomed crop resolves
torso, thigh and arms separately (~16 cm). Why a plane: DA3's relative depth
carries a smooth trend across any crop, as large as a prone person. A plane
through the ground ring removes it; a constant does not. (Both are findings of
the earlier DA3 work on these bags, `~/Depth-Anything-3/box_altitude.py`.)

The relief is in **DA3's own units, not metres**. Looking straight down at
flat ground, relative depth has no observable metric scale. Stage 4 supplies
the scale.

### Stage 3: Gate-1 geolocalisation: pixels to metres (`gate2/geolocator.py`)

Gate-1 geolocates a casualty by casting the ray through a pixel (intrinsics
K), rotating it camera → body → world NED with roll, pitch and heading, and
intersecting it with the ground at the vehicle's height. Gate-2 **imports that
exact class** from `GATE_1_PIPELINE/deployment_gate_1/coordinate_transformer.py`
and uses it to:

* **place every keypoint in 3D.** A keypoint is not on the ground (a shoulder
  may be 0.5 m up), so the ray is stopped at the keypoint's height *h*:
  `P = C + s·r` with `P_down = −h`. With h = 0 this is exactly Gate-1's ground
  intersection. Off-nadir, a raised joint therefore sits radially inward of
  where its ray meets the ground, as it should.
* report the **casualty's GPS** (Gate-1's own `pixel_to_gps` on the mid-hip).
* **track people on the ground.** Between two processed frames the drone moves,
  so a person standing still can shift hundreds of pixels. Their ground position
  in metres barely moves, so the tracker matches on that.

Attitude comes from `/mavros/imu/data`, using the same Euler convention as
Gate-1's `giver_node` (Gate-1 negates pitch inside the transformer). Heading
comes from `/mavros/global_position/compass_hdg` or, failing that, the IMU yaw
converted to a compass heading. Altitude comes from `rel_alt`, or the lidar,
or the local pose. See §4.

### Stage 4: the angle: fusing the three (`gate2/angle.py`)

For every keypoint: **pixel (FlyPose) + height (DA3 × scale) → 3D point (Gate-1 ray)**.

**The metric scale `a` (metres per DA3 unit)** comes from the body itself.
Every limb has a known length as a fraction ρ of standing height S (Drillis &
Contini; `gate2/keypoints.py`). For each visible segment *i* with ground-plane
length *dᵢ* and relief difference *Δrᵢ*, the 3D length `sqrt(dᵢ² + (a·Δrᵢ)²)`
must equal `S·ρᵢ`. A robust least-squares solve finds *a* and *S* together:
flat segments pin S, raised ones fix a. A soft prior keeps S near 1.70 m.
A wrong altitude or focal length scales every *dᵢ*. S absorbs that only while
the implied stature stays near the prior and inside `angle.stature_bounds_m`.
Measured on the synthetic poses, the worst angle error is 13° at −10% scale,
19° at −20%, and 26° at +10% or more. The 90° and 100° poses stay within 3°
and the 120° pose within 9° (table in §9). **Get the altitude and intrinsics right.** The solve is also pulled toward the person's
**track-median scale**. On these bags a person's DA3 scale is stable to about
±7% within a track, while one frame's keypoints are noisy.

**Each segment's rise** (vertical component of the torso and the leg):

1. **From depth.** The rise is the relief change along the segment's interior
   (15–85%, Theil-Sen line fit), not the difference of the two joint samples.
   A joint where a raised segment meets a flat one picks up the raised one's
   height.
2. **Equal depth stays equal.** If DA3 sees no rise above `depth.min_relief_snr`
   × its noise floor (the ground ring's scatter), the measured near-zero rise is
   used as it is. Body proportions **never** invent a rise that the depth did
   not see.
3. **A significant rise on a moderately foreshortened segment.** DA3's sign is
   trusted, and its size is blended with the length-model rise
   `sqrt((S·ρ)² − d²)`. The weight on depth is `(d / S·ρ)²`, so a segment lying
   in the image plane is measured by DA3 and one pointing at the camera by its
   foreshortening. (DA3 under-reads such segments: the lower end is hidden.)
4. **A strongly foreshortened segment** (shorter than `angle.foreshortened_frac`
   = 55% of its expected length) points at the camera, and its lower end is
   hidden under its upper end. DA3 then sees one covering surface, and its
   reading is not this segment's rise. Keypoints truly at equal depth are never
   this short, so this never overrides rule 2. The size comes from the length
   model, and the direction comes from:
   * **torso**: up. For a person on the ground it can only point up.
   * **leg**: the posture prior (below).
5. **Leg under a folded torso** (they overlap in the image by less than
   `angle.overlap_deg`). The leg's relief is really the torso's top surface,
   so its rise comes from its length alone.
6. **Physical limits.** No joint goes below the ground plane. With
   `posture: ground`, the hip joint is capped at `angle.hip_height_ground_m`
   (0.12 m: the pelvis rests on the ground). DA3 sees the surface *over* the
   hip (belly, buttocks, or thighs covering it), and without this cap a leg
   dropping from that surface to the ankle read −54° on a person lying on
   grass. Every other joint keeps its height *relative to the hip*, so
   equal-depth keypoints stay equal.

**Posture prior (`angle.posture`).** From straight above, a leg pointing at
the camera looks the same whether it is drawn up (sitting) or hangs down
(kneeling, standing). `ground` (default, for casualties) decides from the leg
end's surface height relative to the top of the body: drawn up if ≥ 0.55,
flat if ≤ 0.35, blended in between. This ratio needs no metric scale.
`upright` makes such legs hang down (standing or kneeling subjects).

### Stage 5: tracking and smoothing (`gate2/tracker.py`)

Greedy association: on ground distance (≤ `angle.track_world_gate_m`) when
there is GPS, otherwise on box IoU. Each track reports the **median of its last
`angle.smoothing_window` measurements**, using only measurements from the best
method in the window (a 3D-depth reading is never averaged with a fallback
reading). The per-frame value is in the log as `hip_angle_deg`, the smoothed one
as `hip_angle_smoothed_deg`.

---

## 4. Fallbacks

Each stage resolves its own fallbacks. The HUD shows what every stage is
actually running on, and a degraded stage is drawn in orange.

| Stage | Primary | Fallback chain | Notes |
|---|---|---|---|
| ORT provider | TensorRT → CUDA | CPU | the stock `onnxruntime` wheel is CPU-only |
| Pose model | `flypose_h` | `flypose_s` | at load time and if inference raises at run time |
| Detector | FlyPose detector | Ultralytics YOLO/YOLOE (`pose.yolo_weights`) | if the primary fails to load |
| Depth model | DA3 mono-large | base → small → (same on CPU) → disabled | CUDA OOM first halves `tile_res` down to `min_tile_res` |
| Image transport | auto-detected from the topic type | raw `Image`, `CompressedImage`, `FFMPEGPacket` (decoded in-process with PyAV) | then `ros.image_fallback_topics` |
| Image QoS | matches publishers (RELIABLE if they all are) | BEST_EFFORT | a RELIABLE subscriber never matches a BEST_EFFORT publisher |
| Altitude | `/mavros/global_position/rel_alt` | lidar `/scan` (sweep fit × `lidar_range_scale`) → `/mavros/local_position/pose` z → `geo.fixed_altitude_m` → **unscaled** | unscaled: same angle, no metres |
| Heading | `compass_hdg` | IMU yaw → compass heading | the fcb bags have no `compass_hdg` |
| Attitude | IMU, interpolated to the image stamp | camera assumed level | |
| Telemetry clock | header stamp ±`max_age_s` | newest sample by wall-clock arrival (live only, flagged) | for unsynced camera/autopilot clocks |
| Coordinate transformer | Gate-1 checkout | vendored copy `gate2/_gate1_transformer.py` | |
| Angle method | `3d_depth` | `3d_equal_depth` (opt-in `3d_anthro`) | all are 3D vectors |
| Display | OpenCV window | headless (use `--save-video`) | when there is no `$DISPLAY` |

**Angle methods in detail:**

| method | when | how | HUD colour |
|---|---|---|---|
| `3d_depth` | DA3 available, keypoints inside the crop | §3 Stage 4 | green |
| `3d_depth_flat` | as above, but no segment rises above DA3's noise | heights ≈ 0: the planar angle | green |
| `3d_equal_depth` | no depth (disabled, failed, or person not in the top `depth.max_persons`) | every keypoint at the ground plane via the Gate-1 rays; with no telemetry, at one common camera depth through K | yellow |
| `3d_anthro` (opt-in) | no depth | rise guessed from foreshortening, *assuming* torso and leg rise away from the hip | orange |
| `none` | shoulder/hip/ankle (and knee) not visible | no angle; the reason is shown | grey |

The YOLOE fallback detector needs the MobileCLIP text encoder
(`mobileclip2_b.ts`) to set its "person" class. Ultralytics downloads it into
the working directory on first use, and a copy is in this directory. Run from
here, or copy it, so the fallback works offline on the aircraft.

The leg is hip→ankle. When an ankle is not detected, that side falls back to
hip→knee, which is the same direction only when the knee is straight. The
fallback is recorded in `leg_joint` and the notes.

---

## 5. Running it

### Requirements

Everything is already installed on this workstation. `python3 tools/check_env.py`
verifies it.

* ROS 2 Humble (here) or Jazzy (Jetson), with `rclpy`, `rosbag2_py`,
  `ffmpeg_image_transport_msgs` (`sudo apt install ros-$ROS_DISTRO-ffmpeg-image-transport`)
* Python: `numpy`, `opencv-python`, `pyyaml`, `scipy`, `onnxruntime` (or
  `onnxruntime-gpu`), `torch`, `safetensors`, `omegaconf`, `einops`, `addict`
  (for DA3), `av` (PyAV, to decode FFMPEG packets), optionally `ultralytics`
  (YOLO fallback)
* **not** `cv_bridge` (see §10) and **not** `easydict` (Gate-2 provides its own)

### Modes

```bash
# 1) live ROS 2: the aircraft
./run_live.sh [options]                                 # = python3 run_gate2.py live ...

# 2) a bag over ROS 2 through the live path (DDS, QoS, frame dropping as on the drone)
OFFSET=85 RATE=1.0 ./run_bag.sh ~/rosbags/25m_2x_1.5ms [options]

# 3) a bag read directly: every frame, deterministic, no DDS
python3 run_gate2.py bag ~/rosbags/25m_2x_1.5ms --start 85 --duration 30 --stride 2

# 4) any video file or camera index, no telemetry (unscaled 3D, camera assumed level)
python3 run_gate2.py video clip.mp4
```

Mode 3 seeks straight to `--start` (2 s of pre-roll fills the telemetry
buffers) and decodes the camera's HEVC packets in-process. It reads only bag
topics, never the sidecar CSV or mp4. A frame is released only once the bag
clock has passed its stamp by 0.15 s, so the attitude is interpolated, not
extrapolated.

### Options

| option | meaning |
|---|---|
| `--config FILE` | YAML with only the keys you want to change |
| `--set key=value` | override one key (repeatable), e.g. `--set depth.tile_res=504` |
| `--use-sim-time` | live: follow `/clock` (set by `run_bag.sh`) |
| `--start S`, `--duration S` | bag/video: window to process |
| `--stride N` | bag/video: process every N-th frame |
| `--max-frames N` | stop after N processed frames |
| `--realtime` | bag/video: pace to recorded time and drop frames like live |
| `--zoom 1x\|2x` | camera intrinsics profile (`geo.intrinsics=fcb_1x/fcb_2x`) |
| `--pose-model h\|s` | FlyPose pose model |
| `--no-depth` | disable DA3: `3d_equal_depth` only |
| `--no-lidar` | no lidar at all; relative DA3 depth in an attitude-normalised frame (§5, *No-lidar mode*) |
| `--no-display` | headless |
| `--save-video [PATH]` | write the annotated mp4 (default `outputs/<run>.mp4`) |
| `--no-log` | do not write the JSONL log |
| `--save-frames DIR` | where to store frames with detections (default `outputs/<run>_frames/`) |
| `--no-save-frames` | do not store them |
| `--save-raw-frames` | also store the unannotated frame next to each annotated one |
| `--dump DIR` | save each person's keypoints, DA3 relief, crop and telemetry (`.npz`) for offline analysis |

### Keys (all modes, live included)

| key | action |
|---|---|
| **space** or **p** | pause / resume. A banner shows while paused. In live mode the feed keeps arriving and stale frames are dropped, so resuming continues from the newest frame, not a backlog. |
| **n** or **.** | while paused: process exactly one more frame, then pause again |
| **s** | snapshot of the current annotated frame to `outputs/<run>_snapNNNNN.png` |
| **q** or **Esc** | quit |

### Offline analysis tools

```bash
python3 tools/check_env.py                      # environment / fallback preview
python3 tools/replay_dumps.py outputs/dump      # re-run only the angle stage on --dump output (seconds)
python3 tools/dump_sheet.py outputs/dump sheet.jpg f00051_t8 f00069_t11   # crops + relief side by side
python3 tests/test_angle.py                     # synthetic geometry tests (no models needed)
```

`replay_dumps.py` accepts `--set`, so fusion parameters can be tuned on real
detections without re-running FlyPose or DA3.

### Whole-frame depth statistics (`run_depth_stats.py`)

```bash
python3 run_depth_stats.py ~/rosbags/25m_2x_1.5ms                            # whole bag
python3 run_depth_stats.py ~/rosbags/25m_2x_1.5ms --start 60 --duration 20   # a segment
```

This tool is separate from the waist-angle pipeline. It builds each frame's
depth from every lidar return near that frame, and adds DA3's relief only when
held-out lidar returns show that it helps. The full method is in the script's
docstring. In outline:

1. **Every return goes into the world** (ENU), using the IMU attitude (roll,
   pitch, yaw) and the EKF position *at that return's own time*. It uses the
   lidar's lever arm and lidar_map's fitted mount.
2. **Each return is projected into the frame's camera** using the attitude and
   position at the frame's stamp, plus the camera's lever arm. This removes
   both parallaxes: lidar-to-camera offset, and motion between a return and
   the shutter.
3. **A ground plane is fitted** to all returns within ±0.6 s that land on the
   camera footprint (typically ~210 returns from 5 sweep passes). The fit is
   weighted by each return's measured noise and made robust with Tukey
   reweighting. The ground offset may drift linearly in time inside the
   window, which makes the lidar its own altimeter: the EKF's height jitters
   ~5 cm from sweep to sweep. A level-ground slope prior binds only where the
   returns can't see the slope (along-track while hovering).
4. **DA3 draws the heatmap.** It is detrended the way `person_parallax.py`'s
   depth panel is: DA3 over its own Gaussian low-pass, with sigma 100 px. Two
   changes from that panel:
   - a robust quadratic surface is divided out first, so the edges stay clean
   - raised objects are left out of the low-pass, so there is no halo around
     people

   The colour range is the ground's median ± 4 robust sd, eased over time, so
   the grass stays visible when people are in view. 80 px is cropped from the
   edges. The heatmap is **display only**, with no metric scale.
5. **DA3 enters the metric depth only if the lidar confirms it**
   (`--relief-gain auto`). The relief gain is fitted with each sweep pass held
   out in turn. It is used only if it predicts the held-out returns at least
   5% better than the lidar ground alone.

**Why the heatmap is display only.** At 25 m the lidar finds no relationship
between DA3's relief and the ground's height:

- Over 5,694 ground returns the correlation is -0.05. The lidar's spread is
  3.0 cm, DA3's 15.3 cm at gain 1.
- Where the lidar hits people, it measures 0.11-0.50 m. DA3 says +1.5 to
  +5.2 m, 10-15x too tall.

So the texture the grass shows at survey height (here and in the reference
video) is shading, not height. At ~3 m altitude (t = 190 s) the grass blades
in the heatmap are real relief, but even there the lidar ground under them is
flat to 0.6 cm.

Three calibrations run per bag at start-up, each logged:

- **Range scale** (`--lidar-scale auto`): the lidar's nadir range is fitted
  against rel_alt over the climb, with a free intercept. On 25m_2x_1.5ms it
  gives lidar = 1.0618 x rel_alt + 0.33 m. That matches the earlier climb
  measurement (26.90 / 25.37 = 1.060). The old 0.926 came from a fit forced
  through zero, which the lidar's deck reading biased. At 25 m the two differ
  by ~45 cm.
- **Beam-angle lag** (`--angle-lag auto`): the SF45's reported angle lags its
  range sample. Successive sweep passes therefore tilt alternately across the
  track. On 25m_2x_1.5ms the lag is -0.59 deg by sweep direction (-1.38 ms),
  and correcting it takes the ground spread from 6.9 to 3.1 cm.
- **Noise model**: measured on the beams within 25 deg of nadir. On
  25m_2x_1.5ms it is 1.0 cm in range and 0.31 deg in beam angle.

Measured on 25m_2x_1.5ms (11 airborne test frames, plus a whole-bag lidar sweep):

| | |
|---|---|
| held-out ground error, near-nadir returns | 2.1 cm rms, p95 4.5 cm (includes ~1-2 cm return noise) |
| depth vs every in-frame lidar return | 2.5 cm rms, p95 5.3 cm, equal to the returns' own noise frame by frame |
| corner-minus-centre depth, frame to frame | 1.2 cm per frame |
| ground tilt | median 0.13, p95 0.45 deg (the field is level) |
| DA3 relief in the metric depth | rejected on every test frame: it made held-out errors worse (up to 93%) |

**What these numbers do not cover:**

- **Systematic terms that more returns cannot average down:**
  - `geo.lidar_offset_body`: the lidar's lever arm is not measured, and its
    vertical part enters the depth 1:1.
  - `geo.camera_mount_deg`: the camera boresight is not measured. 1 deg moves
    the depth ~13 cm at the frame edge at 25 m.
  - The range scale is only as good as rel_alt over the climb.
- **Objects:** the metric depth is the lidar ground, so a person does not
  change the statistics; they show only in the display heatmap. The lidar does
  hit people now and then (amber rings on the frame, rejected from the ground
  fit).

Output: `outputs/<bag>_depthstats_<time>.mp4` (2880x1080, a fragmented mp4,
playable while it is still being written) with five panels:

- the frame, with the lidar returns
- the DA3 relief heatmap (display only)
- the histogram with the normal curve N(mean, sd) of the metric depth
- the histogram with the normal curve of the **lidar altitude**: one value per
  in-frame return, the aircraft's height above that ground point. Roll, pitch,
  yaw, the mount and the scan angle are taken out, and each value is carried to
  the frame's time. It is drawn against the ground fit's altitude and rel_alt.
  On 25m_2x_1.5ms the returns agree to 3.4 cm sd at 26 m (roll -4.9 deg) and
  1.7 cm at 14 m, with the mean within 2 mm of the ground fit.
- a full-height statistics column: depth, lidar altitude, and the fusion's own
  accuracy readouts and any fallback in use

A `.csv` beside it has the same numbers, one row per frame (`alt_*` columns
are the lidar altitude statistics).

Fallbacks, shown on the frame and in the CSV:

- too few returns: the last plane, held up to 2 s
- no lidar: a level plane at rel_alt
- below 1 m: not measured
- no IMU: level attitude
- no pose: no motion compensation
- DA3 does not load: the lidar ground alone

DA3 at `--res 1008` runs at about 2.2 fps on the RTX 3050.

---

### No-lidar mode (`--no-lidar`)

Add `--no-lidar` to any command:

```bash
python3 run_gate2.py bag ~/rosbags/25m_2x_1.5ms --start 85 --no-lidar
python3 run_gate2.py video clip.mp4 --no-lidar
./run_live.sh --no-lidar
OFFSET=85 ./run_bag.sh ~/rosbags/25m_2x_1.5ms --no-lidar
```

It loads `config/no_lidar.yaml` on top of the defaults and runs the angle
through `gate2/no_lidar.py` (`RelativeWaistAngleEstimator`). Everything else
is the same code: FlyPose, the DA3 crop, Gate-1, tracking, display, saved
frames, ROS topics. Outputs get a `_nolidar` suffix so both modes' results can
sit side by side.

What changes:

* **No lidar at all.** `/scan` is not subscribed live, not read from bags, and
  removed from the altitude chain (`rel_alt → pose → fixed`). In the default
  mode the lidar is only the *second* altitude source; it is never used for
  depth or for DA3's scale.
* **Relative DA3 depth.** Each pixel's DA3 depth is taken as a ratio to the
  fitted ground plane, `q = depth / ground` (ground = 1). A point at fraction
  *q* along a keypoint's ray stands at height `(1 − q) × camera height` above
  a level ground, exactly, at any off-nadir angle. So joint heights are
  dimensionless fractions of the camera height, `k·(1 − q)`. Here *k* is
  DA3's relative gain: 1 if its crop depth were physically consistent. It is
  solved per frame from body proportions and pulled toward the track median,
  and logged as `relative_gain`. Measured median on 25m_2x_1.5ms: **0.021**.
  The zoomed crop exaggerates relief ~50× relative to the true depth ratio,
  which is why the raw ratio cannot be used without *k*.
* **Normalised roll, pitch, yaw.** The attitude is wrapped to canonical ranges
  (roll and pitch to [−180°, 180°) with |pitch| ≤ 90°, using the equivalent
  Euler form if needed; heading to [0°, 360°)). Every ray is rotated by it into
  a level, north-aligned frame, so elevations are against the true horizontal.
  The normalised attitude is logged per frame (`attitude_deg`).
* **Altitude only as a prior.** `rel_alt` / pose (not lidar) is used for the
  GPS fix, to print the stature in metres, and to set the soft stature prior
  in camera-height units (1.70 m ÷ altitude). Without that prior the stature
  floats, limbs FlyPose draws long are absorbed into a bigger person, and *k*
  collapses to 0 on most frames. With no altitude at all, the solve runs
  fully relative.

Measured agreement with the default mode on the same 30 frames of
25m_2x_1.5ms: hip angle median difference 0.3°, 90th percentile 4.4°, max
6.7°. On synthetic poses at the same altitude it reproduces the default mode
to < 0.5° (the geometry is the same up to a uniform scale). Because the
geometry is relative, every angle (the hip angle and both elevations) means
the same thing as in metric mode; only lengths lose their unit.

## 6. Outputs

### On screen

Per person: box (green = 3D depth, yellow = equal depth, orange = proportions),
skeleton, the **hip→shoulder vector (magenta)** and **hip→ankle vector (yellow)**, an arc
at the hip, and a label:

```
#7 hip 113.4 deg  bend 66.6          ← track id, smoothed hip angle, bend
3D depth  conf 0.27  [L 116  R 109]  ← method, confidence, per-side angles
hip-shoulder +14 deg parallel        ← elevation of the torso vector vs the ground
hip-ankle L -4 parallel  R -2 parallel  ← elevation of each leg vector ("hip-knee" on fallback)
torso & leg aligned (straight)       ← only when they form one straight line
stature 1.60 m                       ← solved stature (metric runs only)
28.752953, 77.117175                 ← casualty GPS (Gate-1 pixel_to_gps on the mid-hip)
```

Top left: source, processing rate, per-stage latency, frame age (live), the
telemetry in use with its source, and the status of each stage (orange =
running on a fallback). Top right: the DA3 relief of the most confident
person, with its keypoints.

### JSONL log (`outputs/<run>.jsonl`, one line per person per frame)

| field | meaning |
|---|---|
| `stamp` | image header stamp (s) |
| `track` | person id |
| `bbox`, `det_score` | detection |
| `method` | angle method used (§4) |
| `hip_angle_deg`, `bend_deg` | this frame |
| `hip_angle_smoothed_deg` | track median (what the overlay shows) |
| `sides_deg` | per-side angles `{L, R}` (or `mid`) |
| `angle_2d_deg` | raw pixel angle, reference only |
| `confidence` | keypoint confidence × method weight × fit quality (0–1) |
| `metric`, `stature_m` | whether the geometry is in metres; solved stature in metres (when an altitude is known) |
| `orientation` | per vector (`torso`, `leg_L`, `leg_R`): `elevation_deg`, `from_vertical_deg`, `azimuth_deg`, `relation` (parallel/perpendicular/oblique), `direction`; plus `aligned`, `measured`, `summary` (§2) |
| `elevation_smoothed_deg` | track-median elevation per vector (what the overlay shows) |
| `leg_joint` | `ankle`, or `knee` where the ankle fell back |
| `mode`, `geometry` | `default` / `no_lidar`; `metric`, `unscaled` or `relative` |
| `relative_gain` | no-lidar mode: DA3's relative gain *k* |
| `attitude_deg` | roll, pitch, heading used (normalised in no-lidar mode) |
| `relief_snr` | largest torso/leg relief change ÷ DA3 noise floor |
| `fit_rms` | relative RMS of the limb-length fit |
| `gps`, `altitude_m`, `alt_source` | casualty position, altitude used and its source |
| `keypoints`, `kp_scores` | FlyPose output (pixels) |
| `notes` | every fallback or assumption that applied to this measurement |
| `timings_ms` | pose / depth / angle / total for the frame |

### Detected frames (`outputs/<run>_frames/`)

Every processed frame with at least one detected person is stored as

```
frame_000123_1789186476.039.jpg       annotated frame (as shown on screen)
frame_000123_1789186476.039.json      {"frame", "stamp", "people": [<log records>]}
frame_000123_1789186476.039_raw.jpg   unannotated frame (only with --save-raw-frames)
```

The number is the processed-frame index and the decimal is the image's header
stamp. `output.frames_all: true` stores every processed frame, people or not.
In live mode this is the frames actually processed (~0.7 Hz on this laptop),
not every camera frame.

### ROS topics (live modes)

| topic | type | content |
|---|---|---|
| `/gate2/annotated` | `sensor_msgs/Image` (bgr8) | the annotated frame, stamped with the source image's stamp |
| `/gate2/waist_angles` | `std_msgs/String` | JSON `{"stamp": …, "people": [<log records>]}` |

`std_msgs/String` + JSON avoids needing a custom message package on the
aircraft. Set either topic to `""` in the config to disable it.

---

## 7. Configuration

All settings are in [`config/default.yaml`](config/default.yaml), with a comment
on every key. The ones you are most likely to touch:

| key | default | why you would change it |
|---|---|---|
| `geo.intrinsics` | `fcb_2x` | **must match the camera zoom.** See *Intrinsics* below |
| `ros.image_topic` | `/fcb/image/ffmpeg` | Gate-1's decoded topic is `/fcb/image/decoded`; it is in `image_fallback_topics` |
| `pose.model` | `h` | `s` is ~2× faster on CPU |
| `depth.tile_res` / `tile_res_live` | 1008 / 504 | speed vs depth resolution |
| `depth.max_persons` | 3 | DA3 cost is per person |
| `angle.posture` | `ground` | `upright` for standing/kneeling subjects |
| `angle.use_knee_if_no_ankle` | true | hip→knee fallback when the ankle is not detected |
| `angle.parallel_max_deg` / `perpendicular_min_deg` | 20 / 70 | thresholds for parallel / perpendicular to the ground |
| `output.frames` | `auto` | directory for detected frames (`null` = off) |
| `angle.methods` | `[3d_depth, 3d_equal_depth]` | add `3d_anthro` for a foreshortening guess without depth |
| `geo.altitude_sources` | `[rel_alt, scan, pose, fixed]` | order of altitude sources |
| `geo.fixed_altitude_m` | null | a known flight altitude when no altimeter is available |

### Intrinsics

Metric outputs (stature, GPS, the tracker's ground distances) depend on the
intrinsics. **So does the angle**, for some poses: a focal length that is
wrong by a factor is the same as an altitude wrong by that factor (see the
scale-error table in §9). Profiles:

* `fcb_2x`: the FCB calibration measured in the DA3 work on these bags
  (fx 3168.06 for the full 3840×2160 sensor). The 2x digital zoom is a centred
  1920×1080 crop, so fx is unchanged and the principal point moves to
  (946.6, 497.9).
* `fcb_1x`: same calibration, full frame scaled to the stream (fx ≈ 1584 at 1920 px).
* `gate1`: the matrix hard-coded in GATE_1 `v2/Pipeline_yoloe_logs_robust.py`
  (fx 4335 at 1920×1080).
* `camera_info`: whatever `/fcb/camera_info` says. **On the fcb bags this is a
  placeholder** (fx = fy = 3090, cx, cy = frame centre, zero distortion).
* `custom`: `geo.custom_K`, `custom_K_size`, `custom_K_mode` (crop/scale/asis).

> ⚠️ Gate-1's fx (4335) and the calibration measured for these bags (3168 at
> 2x) disagree by 37%, which is enough to move the angle by up to ~25° on some
> poses (§9). They may describe different optical zoom settings.
> Which one is right for a given flight determines the metric numbers. Check it
> against a known distance on the ground before trusting stature or GPS.

---

## 8. Performance

Measured on this workstation (RTX 3050 6 GB laptop GPU, 16 CPU threads, stock
CPU-only `onnxruntime`), 25m_2x_1.5ms, steady state, ~1.2 people per frame:

| stage | time per frame |
|---|---|
| FlyPose detector + pose, **H**, CPU | 1170 ms |
| FlyPose detector + pose, **S**, CPU | 610 ms |
| DA3 mono-large per person @ 1008 px (CUDA) | 700 ms |
| DA3 mono-large per person @ 504 px (CUDA) | 177 ms (angle within 1.4° median of 1008) |
| geometry + angle fusion | 16 ms |
| **total, live defaults (H, 504)** | **≈ 1.4 s → ~0.7 Hz** |

The bottleneck is FlyPose on the CPU. To go faster, in order of payoff:

1. **`pip install onnxruntime-gpu`** (replacing `onnxruntime`). Gate-2 picks the
   CUDA provider automatically. The FlyPose paper reports ~20 ms inference latency on a
   Jetson AGX Orin. Not done on this machine, because it replaces
   the system-wide `onnxruntime`.
2. `--pose-model s`: ViTPose-S, ~2× faster on CPU, less accurate keypoints.
3. `depth.max_persons: 1`, or a smaller `depth.tile_res_live`.

The live node never queues frames. When inference is slower than the camera,
it processes the newest frame and drops the rest; the HUD shows the age of what
it processed. Only one DA3 process fits on the 6 GB card at a time.

---

## 9. Validation, and what it can and cannot do

### Synthetic geometry (`tests/test_angle.py`: 12 tests, all passing)

3D bodies are posed with a known hip angle and projected through the same
Gate-1 camera model (25 m, rolled and pitched, 3.6 m off-nadir). Each is given
a fake DA3 relief: true height × an unknown scale, plus noise. Results:

| true | 3D depth | equal-depth (no DA3) | raw pixel angle |
|---|---|---|---|
| 180° (lying flat) | 180.0° | 180.0° | 180.0° |
| 150° | 155.7° | 175.6° | 175.6° |
| 120° | 122.3° | 167.7° | 167.7° |
| 90° (sitting up) | 90.1° | 111.8° | 111.6° |
| 60° (folded) | 60.4° | 14.8° | 14.7° |
| standing, `posture=upright` | 179.0° | 179.0° | 179.0° |

The tests also check:

* **Keypoints at equal depth give the exact angle.** This holds within 1° for
  170/140/110° flat bodies, with up to 14° of camera tilt, for a 1.45 m person
  (shorter than the 1.70 m prior), and both without DA3 and with a flat
  noise-only DA3 relief. No rise is invented.
* **Scale error** (wrong altitude or focal length by a factor), worst |error|
  over the 180/150/120/100/90/60° poses:

  | scale | ×0.6 | ×0.7 | ×0.8 | ×0.9 | ×1.0 | ×1.1 | ×1.2 | ×1.3 | ×1.4 |
  |---|---|---|---|---|---|---|---|---|---|
  | worst error | 29° | 24° | 19° | 13° | 6° | 26° | 26° | 26° | 26° |

  The 90° and 100° poses stay within 3° throughout, the 120° pose within 9°. The failures are the
  150° pose, which snaps to its equal-depth value once the limbs look longer
  than the stature bounds allow, and the folded 60° pose.
* With no altitude at all, the unscaled geometry gives the same angle.
* The ankle is the leg end, and the knee is used when ankles are hidden.
* Orientation: lying → torso and legs parallel and aligned; sitting upright →
  torso perpendicular (up), legs parallel; standing → torso up and legs down,
  both perpendicular.

### Real data (25m_2x_1.5ms, 65–165 s)

There is no ground truth for posture in the bags, so these are checked against
what the crops visibly show (`tools/dump_sheet.py`). Values are track medians,
with the hip→ankle leg:

| subject | visible posture | hip angle (3D) | equal-depth | hip→shoulder elevation | hip→ankle elevation |
|---|---|---|---|---|---|
| track 7 | lying on the back, knees bent | ~141° | 109–159° | +16° (parallel) | −12° (parallel) |
| track 8 | sitting, knees hugged up | ~102° | 170–179° ✗ | +80° (perpendicular) | 0° (parallel) |
| tracks 3, 6 | sitting cross-legged, hunched | ~42–46° | 9–65° | +69–74° (oblique–perp.) | +28° |
| track 12 | sitting | ~65° | 23–41° | +83° (perpendicular) | +16° (parallel) |

The torso elevation is what separates sitting from lying: +70° to +85° for
everyone sitting, under 20° for those lying. The equal-depth angle has no
access to it. With the earlier hip→knee leg, the knees-hugged subject read
~41°; with hip→ankle it reads ~102°, because the ankles rest on the ground by
the buttocks. Both are correct for what they measure.

### Known limits

* **Keypoint quality bounds everything.** FlyPose on aerial frames sometimes
  places a knee on the body mass instead of the leg (tracks 5 and 18 above,
  people sitting with legs out). No geometry recovers a joint in the wrong
  place. Confidence and the rolling median soften this; they do not fix it.
* **From straight above, some postures are ambiguous.** Standing upright versus
  hugging the knees, and kneeling versus knees drawn up, project to the same
  image and similar relief. `angle.posture` chooses; the default `ground` suits
  casualties.
* **Small people.** Below ~60 px the pose model often cannot form a
  shoulder–hip–ankle (or knee) chain, and the angle is reported as `none` with the reason.
* **The DA3 scale is per crop.** The relief's shape is reliable (people separate
  from grass at tens of sigma), but its metric scale comes from body proportions
  and the track median, not from a measurement. Motion parallax, as in
  `~/Depth-Anything-3/live_parallax.py`, is the measured alternative, but it
  needs ~8 s of dwell per subject. It is a natural future addition as a scale
  source.

---

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| live node prints `nothing is publishing the camera topic` | Check `ros2 topic list`. With `ros2 bag play`, rosbag2 **silently skips** topics whose message package is missing (one WARN at start-up): install `ros-$ROS_DISTRO-ffmpeg-image-transport`. `run_bag.sh` checks this. |
| `packets arriving but no frame yet` | Joining an HEVC stream mid-way: the decoder waits for the next keyframe (every 0.5 s on the FCB). |
| a node that looks hung, with no error | The classic cause is cv_bridge under NumPy 2 (`_ARRAY_API not found`), raised inside a callback where nobody sees it. Gate-2 does not use cv_bridge, and its callbacks never raise: errors are counted and printed in the periodic health line. |
| `/opt/ros/.../setup.bash: unbound variable` | ROS setup scripts break under `set -u`; the launch scripts relax it while sourcing. |
| HUD: `alt none` / `unscaled` | No altitude source has a fresh sample (`ros.*_topic`, `geo.max_age_s`). The angle is still valid; metres and GPS are not. |
| HUD: `(wall-clock)` after a source | The camera and autopilot clocks disagree by more than `geo.max_age_s`, so telemetry is paired by arrival time. Fix the time sync (Gate-1's `timesync_check.sh`). |
| `principal point outside the frame` | Wrong `geo.intrinsics` profile for this stream size or zoom. |
| very slow | FlyPose is on CPU; see §8. |
| `cannot open bag ... mcap` | `sudo apt install ros-$ROS_DISTRO-rosbag2-storage-mcap` |
| CUDA OOM | Handled automatically (resolution halves, then smaller models). Do not run two DA3 processes on the 6 GB card. |

---

## 11. Code layout and extending it

```
Gate-2_Pipeline/
├── run_gate2.py              CLI: live | bag | video
├── run_live.sh               live launcher (sources ROS, Gate-1's FastDDS profile)
├── run_bag.sh                ros2 bag play --clock + the live node
├── config/default.yaml       every setting, commented
├── config/no_lidar.yaml      layer applied by --no-lidar
├── gate2/
│   ├── config.py             YAML + --set overrides, AttrDict
│   ├── pose_estimator.py     Stage 1: FlyPose (+ provider/model/detector fallbacks)
│   ├── depth_estimator.py    Stage 2: DA3 crop relief (+ OOM/model/device fallbacks)
│   ├── geolocator.py         Stage 3: Gate-1 transformer, intrinsics profiles, lift()
│   ├── _gate1_transformer.py vendored copy of Gate-1's transformer (fallback only)
│   ├── angle.py              Stage 4: 3D keypoints, scale solve, rise fusion, methods
│   ├── no_lidar.py           --no-lidar: relative DA3 depth, normalised attitude (parallel estimator)
│   ├── keypoints.py          COCO-17 layout, body-segment proportions
│   ├── tracker.py            Stage 5: ground/IoU tracking, per-track median & scale
│   ├── telemetry.py          header-stamp buffers, altitude/heading/attitude chains
│   ├── image_codec.py        Image / CompressedImage / FFMPEGPacket → BGR (no cv_bridge)
│   ├── pipeline.py           orchestration, rendering, JSONL records, dumps
│   ├── visualize.py          overlay and HUD
│   ├── utils.py              logging, stage-status registry, timers
│   └── sources/
│       ├── ros_live.py       live ROS 2 source + publishers
│       ├── bag_offline.py    direct rosbag2 reader (seek, keyframes, lookahead)
│       └── video.py          video file / camera
├── tools/                    check_env.py, replay_dumps.py, dump_sheet.py
├── tests/test_angle.py       synthetic geometry tests
└── outputs/                  logs, videos, dumps (created on use)
```

Every stage is a class with one call, and stages share only plain data
(keypoint arrays, a `Relief`, a `Telemetry`). Swapping one out does not touch
the others:

* **Another pose model:** return the same dicts from `PoseEstimator.detect`
  (`bbox`, `det_score`, `keypoints` (17,2), `scores` (17,)) in COCO order.
* **Another depth model:** return a `Relief` from `DepthEstimator.relief`
  (relief in any unit; only its shape is used).
* **A metric scale source** (e.g. parallax): pass it as `scale_prior` to
  `WaistAngleEstimator.estimate`, which already accepts the track median there.
* **Another input:** write a source with `frames()` yielding `(bgr, stamp)` and
  a `hub` (`TelemetryHub`) it fills.
