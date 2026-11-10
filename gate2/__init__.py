"""Gate-2: waist (hip) angle of people seen from a drone.

FlyPose finds people and their keypoints, Depth-Anything-3 gives each person's
relative depth relief, and the Gate-1 coordinate transformer puts every
keypoint on the ground in metres. gate2.angle combines the three into the
angle between torso and thigh.
"""
__version__ = "1.0.0"
