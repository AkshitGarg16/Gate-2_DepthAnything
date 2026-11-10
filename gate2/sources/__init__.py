"""Frame sources. Each yields (frame_bgr, header_stamp) and fills a TelemetryHub.

  ros_live.RosLiveSource      live ROS 2 topics (also `ros2 bag play`)
  bag_offline.BagSource       reads a rosbag2 directly, every frame, deterministic
  video.VideoSource           mp4/avi/camera with no telemetry (2D/unscaled fallback)
"""
