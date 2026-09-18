"""Pick-and-place state machine: the layer that decides whether to act.

WHY A STATE MACHINE AND NOT A SCRIPT
------------------------------------
A script assumes every step works. Manipulation is a sequence of steps that
routinely do not: the object is not visible, the pose is ambiguous, IK fails,
the gripper closes on air. A state machine makes each of those an explicit,
named transition with somewhere to go -- which is the difference between a demo
that works once and a system whose failures you can enumerate.

THE PIPELINE'S CENTRAL DESIGN DECISION LIVES HERE
--------------------------------------------------
Step 3 measured that monocular depth uncertainty is ~Z/s times worse than
lateral, and grows with Z^2. For a 33 mm marker at 0.4 m that is roughly 12x
anisotropy. Reaching open-loop from a single distant observation therefore
means betting the grasp on the WORST-measured axis.

So we do not. The arm moves to a PRE-GRASP pose above the object, re-observes
from close range where the same relative error is far smaller, and only then
descends. That is `SERVO`, and it is the whole reason the task layer exists
rather than a straight line from perception to IK.

The second decision: the last few centimetres happen with the camera looking at
the object from ~15 cm rather than ~40 cm. Depth error scales with Z^2, so
closing the distance by 2.5x improves depth certainty by ~6x -- more than any
detector improvement would.

WHAT WE REFUSE TO ACT ON
------------------------
  - a pose flagged ambiguous (Step 3): re-observe from a different angle
    instead of guessing which of two mirrored poses is real
  - a pose older than `max_pose_age`: acting on stale data moves the arm to
    where the object WAS
  - an IK failure: report it rather than approximating the pose
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from vgp.msgs import (
    GripperCommand, Header, JointState, JointTrajectory, ObjectPoseArray,
)
from vgp.perception.nodes import TOPIC_POSES
from vgp.planning.cartesian import CartesianPlanFailure, plan_cartesian_move
from vgp.planning.trajectory import plan_joint_move
from vgp.robot.arm import Arm
from vgp.robot.grasp import grasp_candidates
from vgp.robot.gripper import approach_width, grasp_width, is_holding, OPEN_WIDTH_M
from vgp.robot.ik import solve_ik_multistart
from vgp.robot.node import TOPIC_GRIPPER, TOPIC_JOINT_STATES, TOPIC_TRAJECTORY
from vgp.runtime.node import Node
from vgp.calib.handeye_collect import look_at
from vgp.utils.frames import BASE, CAMERA_OPTICAL, TOOL
from vgp.utils.transforms import SE3


class State(Enum):
    IDLE = auto()
    SEARCH = auto()        # look for the target
    APPROACH = auto()      # joint-space move to a pre-grasp stand-off
    SERVO = auto()         # re-observe from close range, correct the estimate
    PREGRASP = auto()      # get directly ABOVE the grasp, gripper pointing down
    DESCEND = auto()       # straight-line Cartesian down onto the object
    CLOSE = auto()         # close the gripper
    VERIFY = auto()        # did we actually get it?
    LIFT = auto()          # straight-line Cartesian up
    TRANSPORT = auto()     # joint-space move over the bin
    RELEASE = auto()       # open the gripper
    RETREAT = auto()       # back to home
    DONE = auto()
    FAILED = auto()


@dataclass
class TaskConfig:
    target_label: str                      # e.g. "cube_1"
    object_width: float = 0.05
    object_height: float = 0.05
    place_xy: tuple[float, float] = (0.10, -0.34)
    standoff_m: float = 0.18               # pre-grasp height above the object
    servo_standoff_m: float = 0.15         # where we re-observe from
    # Tilt of the servo viewpoint away from straight-down. NOT cosmetic: a
    # head-on view of a planar marker is the PnP two-fold ambiguity, and the
    # task correctly refuses to act on it. See _view_pose.
    view_tilt_deg: float = 28.0
    view_azim_deg: float = 180.0
    lift_m: float = 0.12
    pregrasp_m: float = 0.10               # height directly above the grasp
    max_pose_age: float = 0.35             # seconds; older poses are refused
    max_rmse_px: float = 1.5
    servo_tolerance_m: float = 0.004       # stop servoing when corrections are small
    max_servo_iters: int = 3
    # False = commit the grasp from the DISTANT search observation, skipping the
    # close-range approach entirely. The genuine open-loop ablation.
    #
    # An earlier version ablated only the servo ITERATION, leaving the
    # approach-to-stand-off in place -- so the task still re-observed from
    # 0.15 m and the "ablation" produced byte-identical results to the control.
    # An ablation that changes nothing measures nothing, and this one looked
    # like a finding until the two medians turned out to match to 16 digits.
    use_servo: bool = True
    settle_time: float = 0.35              # let physics settle after a motion
    grip_time: float = 0.6
    home_q: np.ndarray | None = None
    object_symmetry: int = 4
    # Joint configurations to look from, tried in order. SEARCH is an ACTION
    # for an eye-in-hand camera: the camera only sees where the arm points it.
    scan_qs: tuple = ()
    scan_settle: float = 0.5
    search_timeout_per_pose: float = 1.5
    # Distance from the MEASURED pose to the point we should actually grasp,
    # along the measured frame's -Z. See _do_servo: the marker is on the
    # object's top face, so its pose is NOT the object's centre.
    marker_to_centre_m: float | None = None


@dataclass
class TaskLog:
    state_history: list[tuple[float, str]] = field(default_factory=list)
    servo_corrections_m: list[float] = field(default_factory=list)
    failure_reason: str = ""
    grasp_pose: SE3 | None = None
    final_object_pos: np.ndarray | None = None
    # The object pose the task COMMITTED to, in base_link. Compared against
    # simulator ground truth by the eval harness to separate perception error
    # from control error -- the two are indistinguishable at the gripper.
    committed_estimate: SE3 | None = None


class PickPlaceNode(Node):
    def __init__(self, name, bus, tf, clock, arm: Arm, config: TaskConfig) -> None:
        super().__init__(name, bus, tf, clock)
        self.arm = arm
        self.cfg = config
        self.state = State.IDLE
        self.log_data = TaskLog()
        self._poses: ObjectPoseArray | None = None
        self._joints: JointState | None = None
        self._wait_until = 0.0
        self._servo_iter = 0
        self._target_pose_base: SE3 | None = None
        self._grasp_pose: SE3 | None = None
        self._search_deadline = 0.0

    def on_configure(self) -> None:
        self.declare(TOPIC_TRAJECTORY, JointTrajectory)
        self.declare(TOPIC_GRIPPER, GripperCommand)
        self.subscribe(TOPIC_POSES, ObjectPoseArray, self._on_poses)
        self.subscribe(TOPIC_JOINT_STATES, JointState, self._on_joints)
        self.create_timer(20.0, self._tick, name="task")

    def on_start(self) -> None:
        self._scan_idx = -1
        self._transition(State.SEARCH)
        self._search_deadline = self.now()      # forces an immediate first move
        self._command_gripper(OPEN_WIDTH_M)

    # ------------------------------------------------------------- inputs

    def _on_poses(self, msg: ObjectPoseArray) -> None:
        self._poses = msg

    def _on_joints(self, msg: JointState) -> None:
        self._joints = msg

    def _current_target(self) -> SE3 | None:
        """Latest usable pose of the target, in base_link. None if unusable.

        Every rejection is logged with a REASON. Silent "no target" is the
        least debuggable possible failure -- "we saw it but the pose was 0.4 s
        old" and "we never saw it" need completely different fixes.
        """
        if self._poses is None:
            return None
        now = self.now()
        for obj in self._poses.objects:
            if obj.label != self.cfg.target_label:
                continue
            age = now - obj.header.stamp
            if age > self.cfg.max_pose_age:
                self.log(f"pose of {obj.label} is {age*1000:.0f}ms old; refusing", "warn")
                return None
            if obj.reprojection_rmse_px > self.cfg.max_rmse_px:
                self.log(f"reprojection {obj.reprojection_rmse_px:.2f}px too high", "warn")
                return None
            if obj.pose_ambiguous:
                # Step 3's two-fold planar ambiguity. Acting on the wrong branch
                # tilts the approach axis into the table.
                self.log(f"pose of {obj.label} is ambiguous; need another view", "warn")
                return None
            try:
                # Resolve at the IMAGE's timestamp, not now. This is what the
                # TF buffer's interpolation is for.
                T_base_cam = self.tf.lookup(BASE, obj.header.frame_id, obj.header.stamp)
            except Exception as e:
                self.log(f"TF lookup failed: {e}", "warn")
                return None
            return SE3((T_base_cam @ SE3(obj.pose.matrix)).matrix, BASE, obj.label)
        return None

    # ------------------------------------------------------------ outputs

    def _command_gripper(self, width: float) -> None:
        self.publish(TOPIC_GRIPPER, GripperCommand(Header(self.now(), BASE), width))

    def _send_joint_move(self, q_target, smooth=False, speed=1.0) -> float:
        traj = plan_joint_move(self.arm.q, q_target, self.arm.joint_names,
                               self.now(), smooth=smooth, speed_scale=speed)
        self.publish(TOPIC_TRAJECTORY, traj)
        return traj.duration

    def _send_cartesian_move(self, target: SE3, speed=0.35) -> float | None:
        try:
            traj = plan_cartesian_move(self.arm, self.arm.q, target,
                                       self.arm.joint_names, self.now(),
                                       speed_scale=speed)
        except CartesianPlanFailure as e:
            self._fail(str(e))
            return None
        self.publish(TOPIC_TRAJECTORY, traj)
        return traj.duration

    def _view_pose(self, point, height: float, tilt_deg: float = 0.0,
                   azim_deg: float = 0.0) -> np.ndarray | None:
        """Joints placing the CAMERA above `point`, looking down.

        WHY THIS IS NOT THE SAME AS PUTTING THE GRIPPER ABOVE THE OBJECT.
        With an eye-in-hand mount the camera sits off the tool axis (55 mm
        across and pitched 22 degrees, here). Pose the TOOL above the object
        and the camera is looking roughly 110 mm to one side of it -- so at a
        150 mm stand-off the object is simply out of frame. Measured: the task
        approached correctly and then reported "lost sight of the target while
        servoing", with nothing wrong except that we aimed the wrong frame.

        So we specify the CAMERA pose and convert to a tool pose through the
        hand-eye transform, read from TF rather than passed in -- the robot
        node already publishes it, and having one source avoids the two copies
        disagreeing.

        This is the cost side of eye-in-hand. An eye-to-hand camera sees the
        whole cell all the time; it just gets occluded by the arm at exactly
        the moment you care about.
        """
        try:
            T_tool_cam = self.tf.lookup(TOOL, CAMERA_OPTICAL)
        except Exception as e:
            self.log(f"no hand-eye transform in TF: {e}", "error")
            return None
        # Viewpoint on a cone of half-angle `tilt_deg` above the point.
        #
        # WHY WE DELIBERATELY DO NOT LOOK STRAIGHT DOWN.
        # A planar marker viewed head-on has TWO PnP solutions with nearly
        # identical reprojection error (Step 3). Straight down is the most
        # natural viewpoint for a top-down grasp and the worst possible one for
        # measuring the pose: our ambiguity check fires on every single frame,
        # and rightly refuses to act.
        #
        # Measured in Step 3, the runner-up/best error ratio against tilt:
        #     0 deg -> 1.09 (ambiguous)      15 deg -> 11.2
        #     5 deg -> 2.95 (ambiguous)      30 deg -> 47.8
        # So ~25 degrees of tilt turns an unusable measurement into a
        # comfortable one, for free.
        #
        # The general principle: when an estimator is ill-conditioned, the
        # cheapest fix is usually to change the VIEWPOINT, not the estimator.
        # We control where the camera goes -- that is the advantage of having
        # it on a robot.
        point = np.asarray(point, float)
        tilt, az = np.deg2rad(tilt_deg), np.deg2rad(azim_deg)
        eye = point + height * np.array([
            np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
        T_base_cam = look_at(eye, point, roll=0.0)
        T_base_tool = SE3((T_base_cam @ SE3(T_tool_cam.matrix).inv()).matrix,
                          BASE, TOOL)
        r = solve_ik_multistart(self.arm, T_base_tool, site=self.arm.tool_sid,
                                q_rest=self.cfg.home_q, pos_tol=1e-3, rot_tol=1e-2)
        return r.q if r.success else None

    def _solve_grasp(self, position, yaw) -> tuple[np.ndarray, SE3] | None:
        """IK for a grasp, spending the gripper/object symmetry (Step 5)."""
        for T in grasp_candidates(position, yaw, self.cfg.object_symmetry):
            r = solve_ik_multistart(self.arm, T, q_rest=self.cfg.home_q)
            if r.success:
                return r.q, T
        return None

    # -------------------------------------------------------------- state

    def _transition(self, s: State) -> None:
        self.state = s
        self.log_data.state_history.append((self.now(), s.name))
        self.log(f"-> {s.name}")

    def _wait(self, seconds: float) -> None:
        self._wait_until = self.now() + seconds

    def _fail(self, reason: str) -> None:
        self.log_data.failure_reason = reason
        self.log(f"FAILED: {reason}", "error")
        self._transition(State.FAILED)

    @property
    def finished(self) -> bool:
        return self.state in (State.DONE, State.FAILED)

    def _tick(self) -> None:
        if self.finished or self.now() < self._wait_until:
            return
        getattr(self, f"_do_{self.state.name.lower()}", lambda: None)()

    # --------------------------------------------------------- behaviours

    def _do_search(self) -> None:
        T = self._current_target()
        if T is None:
            if self.now() < self._search_deadline:
                return
            # Move to the next vantage point. For an eye-in-hand camera,
            # searching means MOVING -- the camera sees only where the arm
            # points it.
            self._scan_idx += 1
            if self._scan_idx >= len(self.cfg.scan_qs):
                self._fail(
                    f"never saw a usable pose for {self.cfg.target_label} from "
                    f"{len(self.cfg.scan_qs)} vantage points")
                return
            q = self.cfg.scan_qs[self._scan_idx]
            self.log(f"scanning from vantage {self._scan_idx}")
            d = self._send_joint_move(q, speed=0.8)
            self._wait(d + self.cfg.scan_settle)
            # Measure the search window from when the ARM ARRIVES, not from
            # when the move was commanded. Setting it from `now` gives the
            # camera a window that has already expired by the time it is
            # pointing anywhere useful -- the search then "times out" having
            # looked for zero seconds, and reports that it never saw the
            # object, which is true and completely misleading.
            self._search_deadline = self._wait_until + self.cfg.search_timeout_per_pose
            return
        self._target_pose_base = T
        if not self.cfg.use_servo:
            # ABLATION: commit from this distant view, no close-range look.
            self.log("servo disabled: committing from the search observation", "warn")
            self._command_gripper(approach_width(self.cfg.object_width))
            self._commit_grasp(T)
            return
        q = self._view_pose(T.t, self.cfg.standoff_m,
                            self.cfg.view_tilt_deg, self.cfg.view_azim_deg)
        if q is None:
            self._fail(f"no IK solution viewing {self.cfg.target_label} "
                       f"from {self.cfg.standoff_m*1000:.0f}mm above {np.round(T.t,3)}")
            return
        self._command_gripper(approach_width(self.cfg.object_width))
        self._wait(self._send_joint_move(q) + self.cfg.settle_time)
        self._servo_iter = 0
        self._transition(State.APPROACH)

    def _do_approach(self) -> None:
        self._transition(State.SERVO)

    def _do_servo(self) -> None:
        """Re-observe from close range and correct.

        THE POINT OF THIS STATE. The first observation came from ~0.4 m where
        depth error is worst. From the stand-off the same object is ~0.15 m
        away, and depth error scales with Z^2 -- so the correction applied here
        is typically several millimetres and is the difference between a grasp
        and a nudge.

        We iterate until the correction is small (converged) or we run out of
        attempts (something is wrong and more looking will not fix it).
        """
        T = self._current_target()
        if T is None:
            self._servo_iter += 1
            if self._servo_iter > self.cfg.max_servo_iters:
                self._fail("lost sight of the target while servoing")
            return

        correction = float(np.linalg.norm(T.t - self._target_pose_base.t))
        self.log_data.servo_corrections_m.append(correction)
        self._target_pose_base = T
        self._servo_iter += 1

        if correction > self.cfg.servo_tolerance_m and self._servo_iter <= self.cfg.max_servo_iters:
            q = self._view_pose(T.t, self.cfg.servo_standoff_m,
                                self.cfg.view_tilt_deg, self.cfg.view_azim_deg)
            if q is None:
                self._fail("no IK solution at the servo stand-off")
                return
            self._wait(self._send_joint_move(q, smooth=True, speed=0.6) + self.cfg.settle_time)
            return

        # Converged: commit to a grasp pose.
        self._commit_grasp(T)

    def _commit_grasp(self, T: SE3) -> None:
        yaw = float(np.arctan2(T.R[1, 0], T.R[0, 0]))

        # THE MARKER IS NOT THE OBJECT.
        # Perception measures the pose of the fiducial, which is stuck on the
        # object's TOP FACE. Grasping there puts the fingertips level with the
        # top of the cube -- they close on the last millimetre of it.
        #
        # The symptom was not a failed grasp. It was a grasp that PASSED the
        # holding check (the fingers were genuinely blocked), lifted the cube
        # successfully, and then dropped it partway through the transport move.
        # Diagnosed by logging the object position per state: it rose with the
        # gripper and then reappeared on the table 100 mm from the bin.
        #
        # So we offset along the measured frame's -Z (into the object) by half
        # its height, which puts the TCP at the cube's mid-height where the
        # fingers close around its widest part. The offset is a known property
        # of how the marker was mounted -- part of the object model, not
        # something perception can discover.
        offset = (self.cfg.marker_to_centre_m
                  if self.cfg.marker_to_centre_m is not None
                  else self.cfg.object_height / 2.0)
        grasp_pt = T.t - T.R[:, 2] * offset
        sol = self._solve_grasp(grasp_pt, yaw)
        if sol is None:
            self._fail("no IK solution at the grasp pose")
            return
        _, self._grasp_pose = sol
        self.log_data.grasp_pose = self._grasp_pose
        self.log_data.committed_estimate = SE3(T.matrix, BASE, self.cfg.target_label)
        self._transition(State.PREGRASP)

    def _do_pregrasp(self) -> None:
        """Get the TOOL directly above the grasp before descending.

        WHY THIS STATE HAD TO EXIST.
        We servo from a TILTED viewpoint (to break the planar ambiguity), so
        when SERVO converges the arm is off to one side. A single Cartesian
        move from there to the grasp travels DIAGONALLY -- straight through the
        side of the object it is trying to pick. Measured: the fingers shoved
        the cube 14 mm and forced themselves open to 74 mm.

        Fixing it needs the classic two-stage approach: a free move to a point
        directly above the grasp, then a purely vertical descent. The vertical
        leg is what makes the motion predictable around neighbouring objects,
        and it is why "approach along the tool axis" is the standard pattern
        rather than a stylistic choice.

        This leg is joint-space: it is free-air motion where the path shape
        does not matter, and joint-space is faster and cannot hit a
        singularity mid-path.
        """
        above = SE3.from_rt(self._grasp_pose.R,
                            self._grasp_pose.t + np.array([0, 0, self.cfg.pregrasp_m]),
                            parent=BASE, child="grasp")
        r = solve_ik_multistart(self.arm, above, q_rest=self.cfg.home_q)
        if not r.success:
            self._fail(f"no IK solution directly above the grasp: {r.message}")
            return
        self._wait(self._send_joint_move(r.q, smooth=True, speed=0.6) + self.cfg.settle_time)
        self._transition(State.DESCEND)

    def _do_descend(self) -> None:
        """Straight down, in Cartesian space.

        A joint-space move here would sweep an arc and can drag a fingertip
        through the target or its neighbour. This is the move that justifies
        having a Cartesian planner at all.
        """
        d = self._send_cartesian_move(self._grasp_pose, speed=0.25)
        if d is None:
            return
        self._wait(d + self.cfg.settle_time)
        self._transition(State.CLOSE)

    def _do_close(self) -> None:
        self._command_gripper(grasp_width(self.cfg.object_width))
        self._wait(self.cfg.grip_time)
        self._transition(State.VERIFY)

    def _do_verify(self) -> None:
        """Check the fingers stopped SHORT -- something is between them.

        If they closed fully we are holding air. Detecting that here, before
        lifting, is what turns a silent failure into a reported one.
        """
        w = self.arm.gripper_width
        if not is_holding(w, self.cfg.object_width):
            self._fail(
                f"gripper closed to {w*1000:.1f}mm; expected to be blocked near "
                f"{self.cfg.object_width*1000:.0f}mm -- grasped nothing")
            return
        self.log(f"holding: fingers stopped at {w*1000:.1f}mm")
        self._transition(State.LIFT)

    def _do_lift(self) -> None:
        up = SE3.from_rt(self._grasp_pose.R,
                         self._grasp_pose.t + np.array([0, 0, self.cfg.lift_m]),
                         parent=BASE, child="grasp")
        d = self._send_cartesian_move(up, speed=0.3)
        if d is None:
            return
        self._wait(d + self.cfg.settle_time)
        self._transition(State.TRANSPORT)

    def _do_transport(self) -> None:
        px, py = self.cfg.place_xy
        target = np.array([px, py, self.cfg.object_height / 2 + 0.035])
        sol = self._solve_grasp(target, 0.0)
        if sol is None:
            self._fail(f"no IK solution over the place target {np.round(target,3)}")
            return
        q, T = sol
        self._place_pose = T
        self._wait(self._send_joint_move(q, speed=0.8) + self.cfg.settle_time)
        self._transition(State.RELEASE)

    def _do_release(self) -> None:
        self._command_gripper(OPEN_WIDTH_M)
        self._wait(self.cfg.grip_time)
        self._transition(State.RETREAT)

    def _do_retreat(self) -> None:
        if self.cfg.home_q is not None:
            self._wait(self._send_joint_move(self.cfg.home_q, speed=0.8) + 0.2)
        self._transition(State.DONE)
