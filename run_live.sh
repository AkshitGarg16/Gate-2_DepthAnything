#!/usr/bin/env bash
# run_live.sh -- Gate-2 waist-angle node on the live drone topics.
#
# Usage:
#   ./run_live.sh                         # defaults from config/default.yaml
#   ./run_live.sh --zoom 1x               # any run_gate2.py option passes through
#   ./run_live.sh --no-display --save-video
#   ROS_SETUP=/opt/ros/jazzy/setup.bash ./run_live.sh   # Jetson (Jazzy)
#
# The camera topic may be raw Image, CompressedImage or the FCB's FFMPEGPacket
# stream: the transport is detected from the advertised type and FFMPEG
# packets are decoded in-process, so no image_transport republisher is needed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROS_SETUP=${ROS_SETUP:-}
if [ -z "$ROS_SETUP" ]; then
    for d in jazzy humble; do
        [ -f /opt/ros/$d/setup.bash ] && ROS_SETUP=/opt/ros/$d/setup.bash && break
    done
fi
[ -n "$ROS_SETUP" ] || { echo "[run_live] no ROS 2 install found (set ROS_SETUP)" >&2; exit 1; }
# ROS setup scripts reference unset variables: relax -u while sourcing
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u
# Gate-1's large-message FastDDS profile, if present (6 MB raw frames)
GATE1_DDS="$HOME/GATE_1_PIPELINE/fastdds_large_shm.xml"
if [ -z "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ] && [ -f "$GATE1_DDS" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$GATE1_DDS"
fi
exec python3 "$HERE/run_gate2.py" live "$@"
