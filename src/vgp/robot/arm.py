"""Arm interface over a MuJoCo model: FK, Jacobian, joint state, gripper.

WHY JOINTS ARE ADDRESSED BY NAME AND NEVER BY INDEX
---------------------------------------------------
MuJoCo packs every joint into one flat `qpos` vector in declaration order. Our
scene declares the cubes' free joints BEFORE the arm, so `qpos[0:6]` is a
cube's pose, not the arm's configuration. Slicing by a guessed index gives a
robot that moves confidently to the wrong place, and the bug survives casual
testing because the numbers all look like plausible joint angles.

So every access goes through `mj_name2id` once at construction, and the
resulting index arrays are the only thing used afterwards. Free joints also
occupy 7 qpos entries but only 6 qvel entries (quaternion vs. angular
velocity), so qpos and qvel indices are DIFFERENT and are stored separately.
Conflating them is the classic second version of this bug.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    raise ImportError("MuJoCo required: pip install mujoco")

from vgp.msgs import Header, JointState
from vgp.robot.scene import JOINT_NAMES
from vgp.utils.transforms import SE3


class Arm:
    def __init__(self, model, data, joint_names=JOINT_NAMES,
                 tool_site: str = "tool0_site", grasp_site: str = "grasp_site",
                 finger_joints=("fl", "fr")) -> None:
        self.model, self.data = model, data
        self.joint_names = tuple(joint_names)

        self.jids = np.array([self._jid(n) for n in self.joint_names])
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.qvel_adr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.act_adr = np.array([self._aid(n) for n in ("a1", "a2", "a3", "a4", "a5", "a6")])

        self.finger_jids = np.array([self._jid(n) for n in finger_joints])
        self.finger_qpos = np.array([model.jnt_qposadr[j] for j in self.finger_jids])
        self.finger_act = np.array([self._aid(n) for n in ("afl", "afr")])

        self.tool_sid = self._sid(tool_site)
        self.grasp_sid = self._sid(grasp_site)

        # Limits come FROM the model, never duplicated in Python. A second copy
        # is a second thing to get out of sync.
        self.qlim = model.jnt_range[self.jids].copy()

    # ------------------------------------------------------------- lookup

    def _jid(self, n):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if i < 0: raise ValueError(f"no joint '{n}'")
        return i

    def _aid(self, n):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if i < 0: raise ValueError(f"no actuator '{n}'")
        return i

    def _sid(self, n):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
        if i < 0: raise ValueError(f"no site '{n}'")
        return i

    def _bid(self, n):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        if i < 0: raise ValueError(f"no body '{n}'")
        return i

    # -------------------------------------------------------------- state

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    @q.setter
    def q(self, value) -> None:
        self.data.qpos[self.qpos_adr] = np.asarray(value, dtype=np.float64)

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr].copy()

    @property
    def gripper_width(self) -> float:
        """Distance between fingertips: both fingers slide outward from centre."""
        return float(self.data.qpos[self.finger_qpos].sum())

    def joint_state(self, stamp: float) -> JointState:
        return JointState(Header(stamp, "base_link"), self.joint_names, self.q, self.dq)

    # ----------------------------------------------------------------- FK

    def fk(self, site: int | None = None, q: np.ndarray | None = None) -> SE3:
        """Pose of a site in base_link. Optionally for a hypothetical q.

        When `q` is given we must restore the real state afterwards: FK for
        planning must never perturb the simulation the controller is running
        against. `mj_kinematics` (not `mj_step`) computes poses without
        advancing physics.
        """
        sid = self.tool_sid if site is None else site
        if q is None:
            return self._site_pose(sid)
        saved = self.data.qpos.copy()
        try:
            self.q = q
            mujoco.mj_kinematics(self.model, self.data)
            return self._site_pose(sid)
        finally:
            self.data.qpos[:] = saved
            mujoco.mj_kinematics(self.model, self.data)

    def _site_pose(self, sid: int) -> SE3:
        R = self.data.site_xmat[sid].reshape(3, 3)
        return SE3.from_rt(R, self.data.site_xpos[sid], parent="base_link", child="site")

    def fk_tool(self, q=None) -> SE3:
        return SE3(self.fk(self.tool_sid, q).matrix, "base_link", "tool0")

    def fk_grasp(self, q=None) -> SE3:
        return SE3(self.fk(self.grasp_sid, q).matrix, "base_link", "grasp")

    def body_pose(self, name: str) -> SE3:
        bid = self._bid(name)
        return SE3.from_quat_wxyz(self.data.xquat[bid], self.data.xpos[bid],
                                  parent="world", child=name)

    # ---------------------------------------------------------- Jacobian

    def jacobian(self, site: int | None = None, q: np.ndarray | None = None) -> np.ndarray:
        """6xN spatial Jacobian [linear; angular] for our joints only.

        `mj_jacSite` returns 3 x nv blocks over ALL degrees of freedom -- which
        here includes the cubes' free joints. We slice the columns belonging to
        the arm. Forgetting that gives a 6x20 matrix whose pseudo-inverse
        happily "solves" IK by moving the cubes.
        """
        sid = self.tool_sid if site is None else site
        if q is not None:
            saved = self.data.qpos.copy()
            self.q = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
        J = np.vstack([jacp[:, self.qvel_adr], jacr[:, self.qvel_adr]])
        if q is not None:
            self.data.qpos[:] = saved
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
        return J

    def manipulability(self, q=None, site=None) -> float:
        """Yoshikawa index sqrt(det(J J^T)) -- volume of the velocity ellipsoid.

        Zero at a singularity. We use it as a COST in IK's nullspace rather than
        as a hard constraint, because pushing away from singularities is a soft
        preference that should never prevent reaching a valid pose.
        """
        J = self.jacobian(site, q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    # --------------------------------------------------------- commands

    def command(self, q_target: np.ndarray) -> None:
        self.data.ctrl[self.act_adr] = np.clip(q_target, self.qlim[:, 0], self.qlim[:, 1])

    def command_gripper(self, width: float) -> None:
        """Width between fingertips, in metres. Each finger travels half."""
        half = np.clip(width / 2.0, 0.0, 0.035)
        self.data.ctrl[self.finger_act] = half

    # ------------------------------------------------------- collision

    def _arm_geoms(self) -> set[int]:
        out = set()
        for b in ("base_link", "link_1", "link_2", "link_3", "link_4", "link_5",
                  "link_6", "tool0", "finger_left", "finger_right", "camera_link"):
            try:
                bid = self._bid(b)
            except ValueError:
                continue
            for g in range(self.model.body_geomadr[bid],
                           self.model.body_geomadr[bid] + self.model.body_geomnum[bid]):
                out.add(g)
        return out

    def in_collision(self, q: np.ndarray | None = None,
                     ignore_objects: bool = True) -> bool:
        """Is this configuration self-colliding (or hitting the table)?

        WHY IK MUST CHECK THIS -- it cost us a whole debugging session.
        A configuration can satisfy the pose exactly, respect every joint
        limit, and be perfectly well-conditioned, and still be one the arm
        cannot physically adopt because it passes through itself. We found an
        "observation pose" with j4 = 259 degrees that folded the wrist into the
        forearm: IK reported success, the controller commanded it, and the arm
        simply stopped partway with a 2.6 rad steady-state error.

        The symptom is deeply misleading. Nothing errors. The actuators
        saturate against the contact constraint (measured: actuator_force
        +200 Nm exactly cancelled by qfrc_constraint -199.4), so the robot
        sits still while the controller insists it is commanding correctly.
        The failure then surfaces four layers away as "the camera cannot see
        the object", because the arm is not where anyone thinks it is.

        Widening the wrist joint limits to +/-2pi (realistic for a UR-style
        arm) is what made these wound-up solutions reachable by the solver --
        so the fix is not narrower limits, it is checking collision.
        """
        saved = self.data.qpos.copy()
        try:
            if q is not None:
                self.q = q
            mujoco.mj_forward(self.model, self.data)   # includes collision detection
            arm_geoms = self._arm_geoms()
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                a1, a2 = g1 in arm_geoms, g2 in arm_geoms
                if not (a1 or a2):
                    continue                       # cube on table etc.
                if ignore_objects and not (a1 and a2):
                    # Arm touching a non-arm geom. The table matters; a cube we
                    # are about to grasp does not -- contact with the target is
                    # the POINT, so treating it as a collision would forbid
                    # every grasp.
                    other = g2 if a1 else g1
                    bid = self.model.geom_bodyid[other]
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                    if name.startswith("cube") or name == "bin":
                        continue
                if c.dist < -1e-4:                 # real penetration, not grazing
                    return True
            return False
        finally:
            self.data.qpos[:] = saved
            mujoco.mj_forward(self.model, self.data)

    def in_limits(self, q: np.ndarray, margin: float = 0.0) -> bool:
        return bool(np.all(q >= self.qlim[:, 0] + margin) and
                    np.all(q <= self.qlim[:, 1] - margin))

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.qlim[:, 0], self.qlim[:, 1])
