"""Arm, IK, collision, grasp symmetry, and trajectory tests."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vgp.planning.trajectory import (
    DEFAULT_AMAX, DEFAULT_VMAX, plan_joint_move, quintic, trapezoidal,
)
from vgp.robot.arm import Arm
from vgp.robot.grasp import equivalent_yaws, grasp_candidates, top_down_grasp
from vgp.robot.gripper import approach_width, grasp_width, is_holding
from vgp.robot.ik import pose_error, solve_ik, solve_ik_multistart
from vgp.robot.scene import HOME_Q, OBSERVE_Q, build_xml, default_scene
from vgp.utils.transforms import SE3


@pytest.fixture(scope="module")
def sim():
    m = mujoco.MjModel.from_xml_string(build_xml(default_scene()))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    arm = Arm(m, d)
    arm.q = HOME_Q
    mujoco.mj_forward(m, d)
    return m, d, arm


class TestJointIndexing:
    def test_arm_joints_are_not_the_first_qpos_entries(self, sim):
        """The scene declares cube free joints BEFORE the arm, so qpos[0:6] is
        a cube's pose. Slicing by a guessed index gives a robot that moves
        confidently to the wrong place."""
        _, _, arm = sim
        assert arm.qpos_adr[0] != 0

    def test_qpos_and_qvel_indices_differ(self, sim):
        """Free joints take 7 qpos entries but only 6 qvel -- conflating them
        is the second version of the same bug."""
        _, _, arm = sim
        assert not np.array_equal(arm.qpos_adr, arm.qvel_adr)

    def test_limits_come_from_the_model(self, sim):
        m, _, arm = sim
        assert np.allclose(arm.qlim, m.jnt_range[arm.jids])


class TestForwardKinematics:
    def test_matches_hand_computed_chain(self, sim):
        """Independent check: for rotations about Y only, the TCP's x is
        sum(len_i * sin(cumulative angle_i)). If FK came only from MuJoCo we
        would be checking MuJoCo against itself."""
        from vgp.robot.scene import FLANGE, FOREARM, UPPER_ARM, WRIST_Z
        _, _, arm = sim
        q = np.array([0.0, -0.6, 1.1, -0.5, 0.0, 0.0])
        th = np.cumsum(q[1:4])
        x = (UPPER_ARM * np.sin(th[0]) + FOREARM * np.sin(th[1])
             + (WRIST_Z + FLANGE + 0.069) * np.sin(th[2]))
        assert np.isclose(arm.fk_grasp(q).t[0], x, atol=2e-3)

    def test_fk_does_not_disturb_state(self, sim):
        _, _, arm = sim
        before = arm.q.copy()
        arm.fk_grasp(np.array([0.3, -0.6, 1.1, -0.5, 0.2, 0.1]))
        assert np.allclose(arm.q, before)

    def test_grasp_site_is_beyond_the_flange(self, sim):
        """The TCP sits between the FINGERTIPS, not at the flange -- that is
        what a grasp pose refers to. Measured ALONG THE TOOL AXIS, since at
        HOME the tool points downward and a naive z comparison is inverted."""
        _, _, arm = sim
        tool = arm.fk_tool()
        along = (arm.fk_grasp().t - tool.t) @ tool.R[:, 2]
        assert along > 0.05


class TestJacobian:
    def test_shape_covers_arm_joints_only(self, sim):
        """mj_jacSite returns columns for ALL dofs, including the cubes'. Not
        slicing gives a solver that 'solves' IK by moving the cubes."""
        _, _, arm = sim
        assert arm.jacobian(arm.grasp_sid).shape == (6, 6)

    def test_matches_finite_differences(self, sim):
        _, _, arm = sim
        q = np.array([0.2, 0.5, 0.9, 1.1, 1.4, 0.3])
        J = arm.jacobian(arm.grasp_sid, q)
        eps = 1e-6
        for i in range(6):
            dq = np.zeros(6); dq[i] = eps
            num = (arm.fk_grasp(q + dq).t - arm.fk_grasp(q - dq).t) / (2 * eps)
            assert np.allclose(J[:3, i], num, atol=1e-4)

    def test_wrist_is_singular_at_j5_zero(self, sim):
        """Y-Z-Y wrist: at j5 = 0 the j4 and j6 axes are parallel and a degree
        of freedom is lost. Starting IK there stalls."""
        _, _, arm = sim
        q = np.array([0.0, 0.60, 1.20, 1.34, 0.0, 0.0])
        assert arm.manipulability(q, arm.grasp_sid) < 1e-9

    def test_not_singular_at_home(self, sim):
        _, _, arm = sim
        assert arm.manipulability(HOME_Q, arm.grasp_sid) > 1e-3


class TestCollision:
    def test_home_is_collision_free(self, sim):
        _, _, arm = sim
        assert not arm.in_collision(HOME_Q)

    def test_observe_pose_is_collision_free(self, sim):
        _, _, arm = sim
        assert not arm.in_collision(OBSERVE_Q)

    def test_detects_a_wound_up_wrist(self, sim):
        """REGRESSION: this exact configuration satisfied the pose, respected
        every joint limit, and folded the wrist through the forearm. The arm
        stalled 2.6 rad short with actuator force exactly cancelled by the
        contact constraint -- and NOTHING errored."""
        _, _, arm = sim
        assert arm.in_collision(np.array([0.071382, 0.085819, 1.678426,
                                          4.517453, 3.0702, -3.143059]))

    def test_ik_rejects_colliding_solutions(self, sim):
        _, _, arm = sim
        r = solve_ik_multistart(arm, top_down_grasp((0.28, 0.02, 0.38), 0.0),
                                q_rest=HOME_Q, collision_free=True)
        assert r.success and not arm.in_collision(r.q)


class TestIK:
    TARGETS = [(0.33, 0.12, 0.10), (0.40, -0.10, 0.08), (0.30, 0.0, 0.15),
               (0.45, 0.20, 0.06), (0.10, -0.34, 0.12), (0.25, -0.25, 0.20),
               (0.35, 0.30, 0.08)]

    @pytest.mark.parametrize("t", TARGETS)
    def test_solves_with_symmetry(self, sim, t):
        _, _, arm = sim
        for T in grasp_candidates(t, 0.0, 4):
            r = solve_ik_multistart(arm, T, q_rest=HOME_Q)
            if r.success:
                assert r.pos_err < 1e-3 and r.rot_err < 1e-2
                return
        pytest.fail(f"no grasp candidate solved for {t}")

    def test_pose_error_uses_rotation_vectors(self, sim):
        """Euler differences are not a valid error metric: they wrap, they are
        sequence-dependent, and they explode near gimbal lock."""
        a = SE3.from_euler("xyz", [0.1, 0.2, 0.3], [0, 0, 0])
        b = SE3.from_euler("xyz", [0.1, 0.2, 0.3], [0.1, 0, 0])
        e = pose_error(a, b)
        assert np.allclose(e[:3], [0.1, 0, 0]) and np.linalg.norm(e[3:]) < 1e-9

    def test_reports_failure_honestly(self, sim):
        """Far outside the workspace must FAIL, not return a near-miss as if
        it succeeded."""
        _, _, arm = sim
        r = solve_ik_multistart(arm, top_down_grasp((1.5, 0.0, 0.5), 0.0), q_rest=HOME_Q)
        assert not r.success and r.pos_err > 0.1

    def test_respects_joint_limits(self, sim):
        _, _, arm = sim
        r = solve_ik_multistart(arm, top_down_grasp((0.33, 0.12, 0.10), 0.0), q_rest=HOME_Q)
        assert arm.in_limits(r.q)


class TestGraspSymmetry:
    def test_gripper_is_180_degree_symmetric(self):
        """Swapping which finger is on which side is the same grasp."""
        assert any(abs(abs(y) - np.pi) < 1e-9 or abs(y) < 1e-9
                   for y in equivalent_yaws(0.0, 1))

    def test_square_object_gives_four_yaws(self):
        assert len(equivalent_yaws(0.3, 4)) == 4

    def test_best_conditioned_first(self):
        ys = equivalent_yaws(0.3, 4)
        assert abs(ys[0]) == min(abs(y) for y in ys)

    def test_all_candidates_are_proper_rotations(self):
        for T in grasp_candidates((0.3, 0.1, 0.05), 0.4, 4):
            assert np.isclose(np.linalg.det(T.R), 1.0)

    def test_tool_z_points_down(self):
        assert np.allclose(top_down_grasp((0.3, 0.1, 0.05), 0.7).R[:, 2], [0, 0, -1])

    def test_symmetry_rescues_a_singular_target(self, sim):
        """A top-down grasp at (0.30, 0, 0.15) with yaw = 0 needs a wrist roll
        of exactly pi -- the Y-Z-Y gimbal lock.

        With multistart, joint-limit masking and the nullspace fade, that yaw
        now scrapes through, but only just: it converges at a smallest singular
        value ~2e-4, three orders of magnitude worse-conditioned than the
        equivalent grasp the symmetry offers. Asserting on CONDITIONING rather
        than on success is the honest test -- a solution obtained at the edge
        of a singularity is one the controller will track badly.
        """
        _, _, arm = sim
        single = solve_ik_multistart(arm, top_down_grasp((0.30, 0.0, 0.15), 0.0),
                                     q_rest=HOME_Q)
        best = None
        for T in grasp_candidates((0.30, 0.0, 0.15), 0.0, 4):
            r = solve_ik_multistart(arm, T, q_rest=HOME_Q)
            if r.success and (best is None or r.min_singular_value > best.min_singular_value):
                best = r
        assert best is not None, "symmetry must find a solution"
        assert best.min_singular_value > 50 * max(single.min_singular_value, 1e-12)


class TestGripper:
    def test_commanded_width_is_smaller_than_the_object(self):
        """The difference is what generates contact force. Commanding exactly
        the object width means the fingers only touch it."""
        assert grasp_width(0.05) < 0.05

    def test_approach_width_clears_but_is_not_maximal(self):
        """Fully open fingers sweep a bigger footprint and knock over the
        neighbouring cube on the way down."""
        assert 0.05 < approach_width(0.05) <= 0.070

    def test_holding_detected_when_fingers_are_blocked(self):
        assert is_holding(0.050, 0.05)

    def test_closed_on_air_is_not_holding(self):
        assert not is_holding(grasp_width(0.05), 0.05)


class TestTrajectories:
    Q0 = np.zeros(6)
    Q1 = np.array([1.0, -0.8, 1.2, 0.5, -0.3, 2.0])

    @pytest.mark.parametrize("f", [trapezoidal, quintic])
    def test_reaches_the_endpoint(self, f):
        p = f(self.Q0, self.Q1, DEFAULT_VMAX, DEFAULT_AMAX)
        assert np.allclose(p.sample(p.duration)[0], self.Q1, atol=1e-9)

    @pytest.mark.parametrize("f", [trapezoidal, quintic])
    def test_starts_at_the_start(self, f):
        p = f(self.Q0, self.Q1, DEFAULT_VMAX, DEFAULT_AMAX)
        assert np.allclose(p.sample(0.0)[0], self.Q0, atol=1e-9)

    @pytest.mark.parametrize("f", [trapezoidal, quintic])
    def test_respects_velocity_limits(self, f):
        p = f(self.Q0, self.Q1, DEFAULT_VMAX, DEFAULT_AMAX)
        v = np.array([p.sample(t)[1] for t in np.linspace(0, p.duration, 400)])
        assert np.all(np.abs(v) <= DEFAULT_VMAX * 1.02)

    def test_joints_are_synchronised(self):
        """All joints must start and stop together, or the end-effector traces
        an arbitrary curve through space."""
        p = trapezoidal(self.Q0, self.Q1, DEFAULT_VMAX, DEFAULT_AMAX)
        mid = p.sample(p.duration / 2)[0]
        frac = (mid - self.Q0) / (self.Q1 - self.Q0)
        assert np.allclose(frac, frac[0], atol=1e-9)

    def test_quintic_is_smoother_and_slower(self):
        """The actual trade: smoothness costs time for the same limits."""
        tp = trapezoidal(self.Q0, self.Q1, DEFAULT_VMAX, DEFAULT_AMAX)
        qp = quintic(self.Q0, self.Q1, DEFAULT_VMAX, DEFAULT_AMAX)
        assert qp.duration > tp.duration
        assert np.allclose(qp.sample(0.0)[1], 0.0) and np.allclose(qp.sample(qp.duration)[1], 0.0)

    def test_short_move_uses_a_triangular_profile(self):
        """A move too short to ever reach vmax. The case people forget, which
        produces a negative cruise time and a NaN."""
        p = trapezoidal(np.zeros(6), np.full(6, 0.001), DEFAULT_VMAX, DEFAULT_AMAX)
        assert p.duration > 0 and np.all(np.isfinite(p.sample(p.duration / 2)[0]))

    def test_message_is_sampled_at_the_control_rate(self):
        from vgp.robot.scene import JOINT_NAMES
        m = plan_joint_move(self.Q0, self.Q1, JOINT_NAMES, 0.0, rate_hz=100.0)
        assert len(m.points) == pytest.approx(m.duration * 100, abs=2)
        assert m.points[-1].time_from_start == pytest.approx(m.duration)
