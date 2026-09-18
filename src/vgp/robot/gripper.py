"""Grasp width selection and success checking."""

from __future__ import annotations

import numpy as np

# Fingers close on the object, so the commanded width must be SMALLER than the
# object -- the difference is what generates contact force through the position
# controller's stiffness. Commanding exactly the object width means the fingers
# just touch it, and the cube slips out on the first acceleration.
GRASP_SQUEEZE_M = 0.008
OPEN_WIDTH_M = 0.068
MAX_WIDTH_M = 0.070


def grasp_width(object_width: float) -> float:
    """Commanded fingertip separation for an object of this width."""
    return float(np.clip(object_width - GRASP_SQUEEZE_M, 0.0, MAX_WIDTH_M))


def approach_width(object_width: float, clearance: float = 0.018) -> float:
    """Width to hold while descending: wide enough to clear, not maximal.

    Opening fully would be safer against position error and worse against
    neighbouring objects -- wide fingers sweep a bigger footprint on the way
    down and knock over the cube beside the target. Clearance is the compromise
    and is stated as a parameter rather than buried.
    """
    return float(np.clip(object_width + 2 * clearance, 0.0, MAX_WIDTH_M))


def is_holding(actual_width: float, object_width: float, tol: float = 0.006) -> bool:
    """Did we actually get it?

    The check is that the fingers stopped SHORT of their commanded width -- an
    object is blocking them. If they closed all the way, we missed and are
    holding air. This is the same signal a real gripper's position feedback
    gives, and it is far more reliable than assuming success because the motion
    completed.
    """
    return bool(actual_width > grasp_width(object_width) + tol * 0.5
                and actual_width < object_width + tol)
