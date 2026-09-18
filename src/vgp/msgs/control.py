"""Robot state and command messages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vgp.msgs.header import Header


@dataclass(frozen=True)
class JointState:
    """Measured joint configuration.

    `names` travels with `position` rather than relying on index order. Joint
    ordering differs between the URDF, the MuJoCo model's qpos layout, and the
    vendor SDK, and an off-by-one in joint order is a robot that moves
    confidently to the wrong pose. Naming makes the mapping explicit and lets
    `reorder_to` assert instead of assume.
    """

    header: Header
    names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray | None = None
    effort: np.ndarray | None = None

    def reorder_to(self, names: tuple[str, ...]) -> np.ndarray:
        """Positions in the caller's joint order, raising if a name is absent."""
        idx = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in names if n not in idx]
        if missing:
            raise KeyError(f"JointState is missing joints {missing}; has {self.names}")
        return np.array([self.position[idx[n]] for n in names])


@dataclass(frozen=True)
class JointTrajectoryPoint:
    positions: np.ndarray
    time_from_start: float          # seconds, relative to trajectory start
    velocities: np.ndarray | None = None


@dataclass(frozen=True)
class JointTrajectory:
    """A time-parameterised path through joint space.

    Time-parameterised, not just a list of waypoints: a geometric path says
    where to go, a trajectory says when, and the "when" is what bounds joint
    velocity and acceleration. Handing a controller bare waypoints pushes that
    decision into the controller where it is invisible. See planning/.
    """

    header: Header
    joint_names: tuple[str, ...]
    points: list[JointTrajectoryPoint] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.points[-1].time_from_start if self.points else 0.0


@dataclass(frozen=True)
class GripperCommand:
    """width in metres between fingertips; force in newtons.

    Commanding WIDTH rather than open/closed because grasp width is a property
    of the object we detected, and we know it -- throwing that away and closing
    blindly until stall is slower and risks crushing. `max_effort` is the
    fallback for objects whose width we mis-estimated.
    """

    header: Header
    width: float
    max_effort: float = 20.0
