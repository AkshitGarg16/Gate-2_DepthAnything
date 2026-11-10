#!/usr/bin/env bash
# run_bag.sh -- replay a rosbag over ROS 2 and run the LIVE node against it.
#
# This exercises exactly the code path the drone uses (DDS, QoS, live frame
# dropping), with the bag standing in for the aircraft. To analyse a bag
# frame-by-frame instead, read it directly:  python3 run_gate2.py bag <BAG>
#
# Usage:
#   ./run_bag.sh ~/rosbags/25m_2x_1.5ms            # from the start
#   OFFSET=85 RATE=0.5 ./run_bag.sh <BAG>           # start 85 s in, half speed
#   ./run_bag.sh <BAG> --no-display --save-video    # extra args -> run_gate2.py
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BAG=${1:?usage: run_bag.sh <bag_dir> [run_gate2 options]}
shift
OFFSET=${OFFSET:-0}
RATE=${RATE:-1.0}
[ -d "$BAG" ] || { echo "[run_bag] no such bag: $BAG" >&2; exit 1; }
ROS_SETUP=${ROS_SETUP:-}
if [ -z "$ROS_SETUP" ]; then
    for d in jazzy humble; do
        [ -f /opt/ros/$d/setup.bash ] && ROS_SETUP=/opt/ros/$d/setup.bash && break
    done
fi
# ROS setup scripts reference unset variables: relax -u while sourcing
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

# rosbag2 silently skips topics whose message package is not installed (one
# WARN, then the topic is simply absent). The FCB camera needs this one:
if grep -q "ffmpeg_image_transport_msgs" "$BAG/metadata.yaml" 2>/dev/null && \
   ! ros2 pkg prefix ffmpeg_image_transport_msgs >/dev/null 2>&1; then
    echo "[run_bag] the camera topic is FFMPEGPacket but ffmpeg_image_transport_msgs is" >&2
    echo "          not installed: sudo apt install ros-\$ROS_DISTRO-ffmpeg-image-transport" >&2
    exit 1
fi

PLAY_PID=
cleanup() {
    trap - EXIT INT TERM
    [ -n "$PLAY_PID" ] && kill -INT -- -"$PLAY_PID" 2>/dev/null
    sleep 1
    [ -n "$PLAY_PID" ] && kill -9 -- -"$PLAY_PID" 2>/dev/null
    wait 2>/dev/null
    echo "[run_bag] stopped"
}
trap cleanup EXIT INT TERM

echo "[run_bag] playing $BAG from +${OFFSET}s at ${RATE}x"
# setsid: ros2 bag play gets its own process group so cleanup can stop all of it
setsid ros2 bag play "$BAG" --clock --start-offset "$OFFSET" --rate "$RATE" &
PLAY_PID=$!
sleep 2
# the node runs in the foreground; q / Esc / Ctrl-C tears everything down
python3 "$HERE/run_gate2.py" live --use-sim-time "$@"
