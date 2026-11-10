"""COCO-17 keypoint layout used by FlyPose, and the body segments the angle uses."""

NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear",
         "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
         "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
IDX = {n: i for i, n in enumerate(NAMES)}

L_SHO, R_SHO = IDX["l_shoulder"], IDX["r_shoulder"]
L_ELB, R_ELB = IDX["l_elbow"], IDX["r_elbow"]
L_WRI, R_WRI = IDX["l_wrist"], IDX["r_wrist"]
L_HIP, R_HIP = IDX["l_hip"], IDX["r_hip"]
L_KNE, R_KNE = IDX["l_knee"], IDX["r_knee"]
L_ANK, R_ANK = IDX["l_ankle"], IDX["r_ankle"]

SIDES = {"L": (L_SHO, L_HIP, L_KNE, L_ANK), "R": (R_SHO, R_HIP, R_KNE, R_ANK)}

# Segment lengths as a fraction of standing height (Drillis & Contini 1966,
# as tabulated in Winter, "Biomechanics and Motor Control of Human Movement").
# Used to give DA3's unit-less relief a metric scale; see gate2/angle.py.
SEGMENTS = [
    # (joint a, joint b, length / stature)
    (L_SHO, L_HIP, 0.288), (R_SHO, R_HIP, 0.288),   # torso side: shoulder -> hip
    (L_HIP, L_KNE, 0.245), (R_HIP, R_KNE, 0.245),   # thigh
    (L_KNE, L_ANK, 0.246), (R_KNE, R_ANK, 0.246),   # shank
    (L_SHO, L_ELB, 0.186), (R_SHO, R_ELB, 0.186),   # upper arm
    (L_ELB, L_WRI, 0.146), (R_ELB, R_WRI, 0.146),   # forearm
    (L_SHO, R_SHO, 0.259), (L_HIP, R_HIP, 0.191),   # shoulder width, hip width
]

SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16)]
# BGR, matching FlyPose's own visualisation: head orange, left green, right blue
LINK_COLORS = [(255, 153, 51)] * 7 + [(0, 255, 0)] * 2 + [(0, 128, 255)] * 2 + \
              [(255, 153, 51)] * 3 + [(0, 255, 0)] * 2 + [(0, 128, 255)] * 2
