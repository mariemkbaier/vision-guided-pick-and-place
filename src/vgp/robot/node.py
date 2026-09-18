"""Robot node: steps physics, executes trajectories, publishes state and TF.

WHY THIS NODE OWNS THE PHYSICS STEP
-----------------------------------
Exactly one thing in the system may advance simulated time, or the components
disagree about what "now" means. Making that explicit -- the robot node steps,
everyone else observes -- mirrors reality, where the robot's controller runs on
its own clock and everything else samples it.

WHY IT PUBLISHES TF RATHER THAN LETTING CONSUMERS ASK
-----------------------------------------------------
base_link -> tool0 -> camera_link is the chain that makes a camera observation
mean something in robot coordinates, and it changes every control cycle. The
robot is the only component that knows it, so it publishes it, time-stamped.
Everything else looks it up at the stamp it cares about. That is what lets
perception work while the arm is moving: a pose measured 80 ms ago resolves
against where the arm WAS, not where it is.

The hand-eye result enters here, as the static tool0 -> camera_link transform.
One number from Step 4, inserted once, and the entire pipeline becomes
metrically correct -- or, if it is wrong, uniformly and confusingly wrong.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    raise ImportError("MuJoCo required")

from vgp.msgs import GripperCommand, Header, JointState, JointTrajectory
from vgp.robot.arm import Arm
from vgp.runtime.node import Node
from vgp.utils.frames import BASE, CAMERA_LINK, CAMERA_OPTICAL, GRASP, TOOL, WORLD
from vgp.utils.frames import T_CAMERA_LINK_FROM_OPTICAL
from vgp.utils.transforms import SE3

TOPIC_JOINT_STATES = "/robot/joint_states"
TOPIC_TRAJECTORY = "/robot/trajectory"
TOPIC_GRIPPER = "/robot/gripper"


class RobotNode(Node):
    def __init__(self, name, bus, tf, clock, model, data, arm: Arm,
                 T_tool_camera_optical: SE3, control_hz: float = 100.0,
                 physics_substeps: int | None = None,
                 gravity_compensation: bool = True) -> None:
        """`T_tool_camera_optical` is the hand-eye result: tool0 -> camera_OPTICAL.

        WHY THE PARAMETER SAYS "OPTICAL" IN ITS NAME.
        Hand-eye calibration necessarily produces the OPTICAL frame, because
        that is the frame PnP works in. But the TF chain we publish is

            tool0 -> camera_link -> camera_optical

        with the second hop being the fixed robotics/vision axis convention
        from utils/frames.py. Inserting the hand-eye result as the FIRST hop
        applies the optical rotation twice.

        It does not error. It produces object positions that are plausible --
        right order of magnitude, wrong place. Ours came out at
        (0.73, -0.14, 0.66) for a cube actually at (0.33, 0.12, 0.03), and the
        only symptom was "no IK solution", four layers from the cause.

        So we convert here, explicitly, once:

            T_tool_link = T_tool_optical @ (T_link_optical)^-1

        and let SE3's frame labels check the composition rather than
        overriding them with hand-written parent/child arguments -- which is
        how this slipped through in the first place.
        """
        super().__init__(name, bus, tf, clock)
        self.model, self.data, self.arm = model, data, arm
        self.T_tool_camera_optical = SE3(T_tool_camera_optical.matrix,
                                         parent=TOOL, child=CAMERA_OPTICAL)
        # Frames are labelled, so a mistake here raises instead of misplacing
        # every object by 40 cm.
        self.T_tool_camera_link = SE3(
            (self.T_tool_camera_optical @ T_CAMERA_LINK_FROM_OPTICAL.inv()).matrix,
            parent=TOOL, child=CAMERA_LINK)
        self.control_hz = control_hz
        self.gravity_compensation = gravity_compensation
        # Physics runs faster than control -- a 2 ms timestep under a 100 Hz
        # controller means 5 substeps. Coupling them would either make control
        # absurdly fast or physics unstable.
        self.substeps = physics_substeps or max(
            1, int(round((1.0 / control_hz) / model.opt.timestep)))
        self._traj: JointTrajectory | None = None
        self._traj_t0 = 0.0
        self.trajectories_executed = 0

    def on_configure(self) -> None:
        self.declare(TOPIC_JOINT_STATES, JointState)
        self.subscribe(TOPIC_TRAJECTORY, JointTrajectory, self._on_traj)
        self.subscribe(TOPIC_GRIPPER, GripperCommand, self._on_gripper)
        self.create_timer(self.control_hz, self._tick, name="control")

    def on_start(self) -> None:
        # Static transforms, inserted once.
        self.tf.set_transform(SE3.identity(WORLD, BASE), static=True)
        self.tf.set_transform(self.T_tool_camera_link, static=True)
        self.tf.set_transform(T_CAMERA_LINK_FROM_OPTICAL, static=True)
        self._publish_state()

    # ------------------------------------------------------------ commands

    def _on_traj(self, traj: JointTrajectory) -> None:
        self._traj = traj
        self._traj_t0 = self.now()
        self.trajectories_executed += 1

    def _on_gripper(self, cmd: GripperCommand) -> None:
        self.arm.command_gripper(cmd.width)

    @property
    def busy(self) -> bool:
        return self._traj is not None

    # -------------------------------------------------------------- update

    def _tick(self) -> None:
        if self._traj is not None:
            t = self.now() - self._traj_t0
            pts = self._traj.points
            if t >= self._traj.duration:
                self.arm.command(pts[-1].positions)
                self._traj = None
            else:
                # Linear interpolation between the sampled points. They were
                # generated AT the control rate, so this is a no-op in practice
                # -- it exists so a trajectory from another source at a
                # different rate still executes correctly.
                i = min(int(t * len(pts) / max(self._traj.duration, 1e-9)), len(pts) - 2)
                while i + 1 < len(pts) - 1 and pts[i + 1].time_from_start < t:
                    i += 1
                a, b = pts[i], pts[i + 1]
                span = max(b.time_from_start - a.time_from_start, 1e-9)
                w = np.clip((t - a.time_from_start) / span, 0, 1)
                self.arm.command(a.positions + w * (b.positions - a.positions))

        for _ in range(self.substeps):
            if self.gravity_compensation:
                # FEEDFORWARD of the gravity and Coriolis terms -- what every
                # real industrial controller does, and what a bare position
                # servo does not.
                #
                # Without it the joint servo must generate holding torque from
                # POSITION ERROR alone: the arm has to sag until kp * error
                # balances gravity. Measured on this arm: 6.8 mm of TCP droop
                # at a typical grasp configuration.
                #
                # 6.8 mm sounds small until you notice the gripper has ~10 mm of
                # clearance per side around a 50 mm cube. The fingers then
                # descend off-centre, catch the cube's edge and shove it aside
                # instead of closing around it -- which is exactly what we saw,
                # and it presents as a perception or calibration error rather
                # than a control one.
                #
                # qfrc_bias is the full inverse-dynamics bias (gravity +
                # Coriolis + centrifugal) at the CURRENT state, so this also
                # compensates the velocity-dependent terms during fast moves.
                # Applied to the arm's DOFs only -- applying it to the cubes'
                # free joints would make them hover.
                self.data.qfrc_applied[self.arm.qvel_adr] = \
                    self.data.qfrc_bias[self.arm.qvel_adr]
            mujoco.mj_step(self.model, self.data)
        self._publish_state()

    def _publish_state(self) -> None:
        stamp = self.now()
        self.publish(TOPIC_JOINT_STATES, self.arm.joint_state(stamp))
        # TF stamped with the control time. Perception will ask for these at
        # ITS image timestamps, and the buffer interpolates between them.
        self.tf.set_transform(self.arm.fk_tool(), stamp=stamp)
        self.tf.set_transform(
            SE3(self.arm.fk_grasp().matrix, parent=BASE, child=GRASP), stamp=stamp)
