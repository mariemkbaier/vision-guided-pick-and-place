"""Hand-eye calibration tests.

The headline property: with a KNOWN X, the solver must recover it exactly from
exact data -- and must be caught by a gate when the data cannot determine it.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from vgp.calib.handeye import HandEyeCalibrator, HandEyeError, solve_park
from vgp.calib.handeye_collect import look_at
from vgp.utils.transforms import SE3

X_TRUE = SE3.from_rt(R.from_euler("xyz", [0.3, -0.45, 0.9]).as_matrix(),
                     [0.055, 0.01, -0.02], "tool0", "camera_optical")
T_BASE_TARGET = SE3.from_rt(R.from_euler("xyz", [0.02, 0.01, 0.6]).as_matrix(),
                            [0.36, 0.0, 0.0])


def _trial(n, axes, noise_t=0.0, noise_r=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cal = HandEyeCalibrator()
    for i in range(n):
        ax = np.array(axes[i % len(axes)], float)
        A = SE3.from_rt(R.from_rotvec(ax / np.linalg.norm(ax) * rng.uniform(0.25, 0.9)).as_matrix(),
                        rng.uniform(-0.12, 0.12, 3) + np.array([0.3, 0, 0.4]))
        B = SE3((X_TRUE.inv() @ SE3(A.matrix).inv() @ T_BASE_TARGET).matrix)
        if noise_t or noise_r:
            B = SE3.from_rt(R.from_rotvec(rng.normal(0, noise_r, 3)).as_matrix() @ B.R,
                            B.t + rng.normal(0, noise_t, 3))
        cal.add(A, B)
    return cal


class TestExactRecovery:
    def test_recovers_x_exactly_from_exact_data(self):
        res = _trial(14, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]).solve()
        dt, dr = X_TRUE.distance_to(res.X)
        assert dt < 1e-9 and dr < 1e-9, f"{dt*1e6:.3f} um, {np.degrees(dr)*1e6:.3f} udeg"

    def test_our_solver_agrees_with_opencv(self):
        """Two INDEPENDENT implementations. Agreement is a real check; a single
        implementation agreeing with itself is not."""
        res = _trial(14, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]).solve()
        assert res.cross_check_t_mm < 1e-6

    def test_residual_is_zero_for_exact_data(self):
        assert _trial(14, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]).solve().residual_t_mm < 1e-6

    def test_all_gates_pass(self):
        assert _trial(14, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]).solve().ok


class TestRelativeMotionFormula:
    """REGRESSION for a derivation error that OpenCV caught for us.

    From A_j X B_j = A_i X B_i:   (A_j^-1 A_i) X = X (B_j B_i^-1)

    Note the asymmetry -- A composes as (older)^-1 (newer) while B composes as
    (older)(newer)^-1. Writing B the same way round as A yields B^-1 and an X
    wrong by ~180 degrees. It was found because our answer disagreed with
    OpenCV's by exactly the amount our answer was wrong.
    """

    def test_motion_pairs_satisfy_ax_equals_xb(self):
        cal = _trial(10, [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        for A, B in cal._motions():
            lhs = (SE3(A.matrix) @ SE3(X_TRUE.matrix)).as_matrix()
            rhs = (SE3(X_TRUE.matrix) @ SE3(B.matrix)).as_matrix()
            assert np.allclose(lhs, rhs, atol=1e-9), "AX = XB must hold for the true X"


class TestIdentifiability:
    """Rotations about ONE axis leave a one-parameter family of solutions.

    The solver does not fail -- it returns a family member with a fine-looking
    residual. This is the Step 2 lesson again: a residual cannot tell you the
    problem was well-posed.
    """

    def test_single_axis_is_rank_deficient(self):
        with pytest.raises(HandEyeError, match="rank-deficient"):
            _trial(12, [(0, 0, 1)]).solve()

    def test_near_single_axis_gives_a_wrong_answer(self):
        res = _trial(14, [(0.02, 0.01, 1), (0, 0.02, 1), (0.01, 0, 1)], 3e-4, 1e-3).solve()
        dt, _ = X_TRUE.distance_to(res.X)
        assert dt * 1000 > 20, "expected a badly wrong X from near-degenerate motions"

    def test_axis_spread_gate_catches_it(self):
        res = _trial(14, [(0.02, 0.01, 1), (0, 0.02, 1), (0.01, 0, 1)], 3e-4, 1e-3).solve()
        assert any("axes span" in p for p in res.problems())

    def test_cross_check_also_catches_it(self):
        """Two methods disagreeing means the DATA is ambiguous, not that one
        implementation is buggy -- an independent signal from the same cause."""
        res = _trial(14, [(0.02, 0.01, 1), (0, 0.02, 1), (0.01, 0, 1)], 3e-4, 1e-3).solve()
        assert res.cross_check_t_mm > 1.0

    def test_good_axes_pass_the_spread_gate(self):
        assert _trial(14, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]).solve().rotation_axis_spread > 30


class TestNoiseScaling:
    @pytest.mark.parametrize("nt,nr,max_mm", [(0.0005, 0.002, 2.0), (0.002, 0.008, 7.0)])
    def test_error_stays_proportionate(self, nt, nr, max_mm):
        errs = []
        for s in range(4):
            res = _trial(16, [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)],
                         nt, nr, seed=s).solve()
            errs.append(X_TRUE.distance_to(res.X)[0] * 1000)
        assert np.mean(errs) < max_mm


class TestGuards:
    def test_needs_enough_poses(self):
        cal = HandEyeCalibrator()
        cal.add(SE3.identity(), SE3.identity())
        with pytest.raises(HandEyeError, match=">= 3 poses"):
            cal.solve()

    def test_solve_park_needs_two_motions(self):
        with pytest.raises(HandEyeError, match=">= 2 motion"):
            solve_park([(SE3.identity(), SE3.identity())])


class TestLookAt:
    def test_optical_z_points_at_the_target(self):
        """Optical convention: +Z toward the scene. A look-at helper that
        assumes OpenGL (-Z forward) is off by 180 degrees."""
        eye, tgt = np.array([0.2, 0.1, 0.5]), np.array([0.36, 0.0, 0.0])
        T = look_at(eye, tgt)
        want = (tgt - eye) / np.linalg.norm(tgt - eye)
        assert np.allclose(T.R[:, 2], want, atol=1e-9)

    def test_is_a_proper_rotation(self):
        T = look_at(np.array([0.2, 0.1, 0.5]), np.array([0.36, 0.0, 0.0]))
        assert np.isclose(np.linalg.det(T.R), 1.0)

    def test_handles_straight_down(self):
        """Degenerate case: the default up-hint is parallel to the view axis."""
        T = look_at(np.array([0.3, 0.0, 0.4]), np.array([0.3, 0.0, 0.0]))
        assert np.isclose(np.linalg.det(T.R), 1.0)
        assert np.allclose(T.R[:, 2], [0, 0, -1], atol=1e-9)
