"""Canonical frame names and the one convention conversion that always bites.

Frame names are string constants, not free-form strings, so a typo is an
AttributeError at import time instead of a TFTree lookup that silently finds
nothing at 30 Hz.

THE FRAME TREE
--------------
    world
      └── base_link                 robot base, Z up, X forward   (REP-103)
            ├── link_0 .. link_6    kinematic chain
            │     └── tool0         flange
            │           └── grasp   TCP, between the fingertips
            │                 └── camera_link      [eye-in-hand mount]
            │                       └── camera_optical
            └── camera_link         [eye-to-hand mount, static]
                  └── camera_optical
    (object poses hang off camera_optical when observed, base_link once fused)

Note camera_link appears in exactly one of two places depending on mount style,
and NOTHING else in the tree changes. That is the payoff of a transform tree:
switching eye-in-hand to eye-to-hand is one re-parent, not a code change in
perception or planning.

THE CONVENTION THAT ALWAYS BITES: camera_link vs camera_optical
--------------------------------------------------------------
These are the SAME physical point with DIFFERENT axis conventions, and every
vision-robotics stack has both:

    camera_link     (robotics, REP-103)   X forward, Y left,  Z up
    camera_optical  (vision,  OpenCV)     Z forward, X right, Y down

OpenCV -- solvePnP, projectPoints, calibrateCamera, the whole library -- works
exclusively in the optical convention. The robot's URDF and every planner work
in the robotics convention. So a fixed -90 deg / +90 deg rotation sits between
them, forever, and it is NOT identity.

Symptom when you forget it: the estimated object pose is a plausible distance
away but rotated ~90 deg, so the arm reaches to a point that is off by roughly
the object's depth, in a direction that changes as you move the camera. People
lose days to this and start "correcting" it with a fudge factor. We make it an
explicit, named, tested constant instead.
"""

from __future__ import annotations

import numpy as np

from vgp.utils.transforms import SE3

# --------------------------------------------------------------------- names

WORLD = "world"
BASE = "base_link"
TOOL = "tool0"          # flange: where the URDF chain ends
GRASP = "grasp"         # TCP: between the fingertips -- what we actually place
CAMERA_LINK = "camera_link"
CAMERA_OPTICAL = "camera_optical"
TABLE = "table"


def link(i: int) -> str:
    """Name of the i-th kinematic link."""
    return f"link_{i}"


def object_frame(object_id: str) -> str:
    """Namespaced object frame, so two detections never collide in the tree."""
    return f"obj/{object_id}"


def tag_frame(tag_id: int) -> str:
    return f"tag/{tag_id}"


# ----------------------------------------------------- the optical rotation

def _optical_from_link() -> np.ndarray:
    """Rotation mapping camera_link axes -> camera_optical axes.

    Derive it by asking where each ROBOTICS axis lands in OPTICAL coordinates.
    Columns of the matrix are exactly that:

        link  X (forward) -> optical  Z (forward)   -> column 0 = (0, 0, 1)
        link  Y (left)    -> optical -X (left=-right)-> column 1 = (-1, 0, 0)
        link  Z (up)      -> optical -Y (up = -down) -> column 2 = (0, -1, 0)

    Stacking those columns:

            [  0  -1   0 ]
            [  0   0  -1 ]
            [  1   0   0 ]

    det = +1 (a rotation, not a reflection) -- SE3's constructor enforces that,
    which is the check that catches you if you transcribe this wrong.
    """
    return np.array(
        [[0.0, -1.0, 0.0],
         [0.0, 0.0, -1.0],
         [1.0, 0.0, 0.0]]
    )


# T_link_optical: maps a point expressed in OPTICAL coords into LINK coords.
# This is the direction you need after solvePnP: PnP gives you the object in
# optical coords, and you want it in the robot's world. Hence .T of the above
# (inverse of a rotation), wrapped so the frame labels are right.
T_CAMERA_LINK_FROM_OPTICAL = SE3.from_rt(
    _optical_from_link().T, np.zeros(3), parent=CAMERA_LINK, child=CAMERA_OPTICAL
)

# The other direction, for projecting a robot-frame point into the image.
T_CAMERA_OPTICAL_FROM_LINK = T_CAMERA_LINK_FROM_OPTICAL.inv()
