"""Geometry tests.

These are property tests, not example tests, on purpose. A transform library
that happens to be right for one hand-picked rotation is worthless -- the bugs
live in sign conventions and axis orders that only show up for general
rotations. So we hammer them with randoms and assert invariants that must hold
for every rigid transform.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from vgp.utils.transforms import SE3, FrameMismatch, interpolate
from vgp.utils.frames import (
    T_CAMERA_LINK_FROM_OPTICAL, T_CAMERA_OPTICAL_FROM_LINK,
)

rng = np.random.default_rng(0)


def random_se3(parent=None, child=None):
    return SE3.from_rt(R.random(random_state=int(rng.integers(1 << 30))).as_matrix(),
                       rng.normal(size=3), parent, child)


class TestInvariants:
    def test_inverse_round_trip(self):
        for _ in range(200):
            T = random_se3()
            I = T @ T.inv()
            assert np.allclose(I.as_matrix(), np.eye(4), atol=1e-9)

    def test_associativity(self):
        """(AB)C == A(BC) -- guards against a transposed composition order."""
        for _ in range(100):
            a, b, c = random_se3(), random_se3(), random_se3()
            assert np.allclose(((a @ b) @ c).as_matrix(), (a @ (b @ c)).as_matrix(), atol=1e-9)

    def test_composition_matches_sequential_apply(self):
        """The whole 'T_a_b @ T_b_c' claim, checked on actual points.

        If our direction convention were backwards, this is the test that fails.
        """
        for _ in range(100):
            T_ab, T_bc = random_se3("a", "b"), random_se3("b", "c")
            p_c = rng.normal(size=(7, 3))
            direct = (T_ab @ T_bc).apply(p_c)
            stepwise = T_ab.apply(T_bc.apply(p_c))
            assert np.allclose(direct, stepwise, atol=1e-9)

    def test_rigid_motion_preserves_distances(self):
        """A rigid transform is an isometry -- catches any scale creeping in."""
        for _ in range(100):
            T = random_se3()
            p = rng.normal(size=(10, 3))
            d0 = np.linalg.norm(p[:, None] - p[None], axis=-1)
            d1 = np.linalg.norm(T.apply(p)[:, None] - T.apply(p)[None], axis=-1)
            assert np.allclose(d0, d1, atol=1e-9)

    def test_translation_column_is_child_origin_in_parent(self):
        """T_a_b.t IS b's origin expressed in a. Both readings of T_a_b agree."""
        for _ in range(50):
            T = random_se3("a", "b")
            assert np.allclose(T.apply(np.zeros(3)), T.t, atol=1e-12)


class TestQuaternionConventions:
    def test_xyzw_wxyz_are_not_silently_interchangeable(self):
        """The bug this whole naming scheme exists to prevent."""
        q_xyzw = R.random(random_state=7).as_quat()
        a = SE3.from_quat_xyzw(q_xyzw, np.zeros(3))
        b = SE3.from_quat_wxyz(q_xyzw, np.zeros(3))   # same numbers, wrong order
        assert not np.allclose(a.R, b.R), "orders must give different rotations"

    def test_quat_round_trips_both_orders(self):
        for _ in range(100):
            T = random_se3()
            assert np.allclose(SE3.from_quat_xyzw(T.quat_xyzw(), T.t).R, T.R, atol=1e-9)
            assert np.allclose(SE3.from_quat_wxyz(T.quat_wxyz(), T.t).R, T.R, atol=1e-9)


class TestValidation:
    def test_rejects_non_orthonormal(self):
        m = np.eye(4); m[:3, :3] *= 2.0        # a scale, not a rotation
        with pytest.raises(ValueError, match="orthonormal"):
            SE3(m)

    def test_rejects_reflection(self):
        """det = -1: a mirrored frame. Would flip the gripper approach side."""
        m = np.eye(4); m[0, 0] = -1.0
        with pytest.raises(ValueError, match="reflection"):
            SE3(m)

    def test_frame_mismatch_refuses_compose(self):
        with pytest.raises(FrameMismatch, match="inner frames"):
            random_se3("base", "cam") @ random_se3("tag", "cam")

    def test_unlabelled_composes_freely(self):
        (random_se3("base", "cam") @ random_se3()).as_matrix()

    def test_inverse_swaps_labels(self):
        T = random_se3("base", "cam").inv()
        assert (T.parent, T.child) == ("cam", "base")


class TestOpticalConvention:
    """The camera_link <-> camera_optical rotation, checked axis by axis.

    Getting this wrong is the single most common bug in vision-robotics stacks,
    so it gets explicit per-axis assertions rather than a round-trip check that
    would pass for the inverse rotation too.
    """

    @pytest.mark.parametrize("p_optical, expected_link, why", [
        ([0, 0, 1], [1, 0, 0],  "optical +Z (forward) -> link +X (forward)"),
        ([1, 0, 0], [0, -1, 0], "optical +X (right)   -> link -Y (right = -left)"),
        ([0, 1, 0], [0, 0, -1], "optical +Y (down)    -> link -Z (down = -up)"),
    ])
    def test_axis_mapping(self, p_optical, expected_link, why):
        got = T_CAMERA_LINK_FROM_OPTICAL.apply(np.array(p_optical, float))
        assert np.allclose(got, expected_link, atol=1e-12), why

    def test_not_identity(self):
        """Guards the 'I'll just skip the conversion' failure mode."""
        assert not np.allclose(T_CAMERA_LINK_FROM_OPTICAL.R, np.eye(3))

    def test_round_trip(self):
        I = T_CAMERA_LINK_FROM_OPTICAL @ T_CAMERA_OPTICAL_FROM_LINK
        assert np.allclose(I.as_matrix(), np.eye(4), atol=1e-12)

    def test_is_a_proper_rotation(self):
        assert np.isclose(np.linalg.det(T_CAMERA_LINK_FROM_OPTICAL.R), 1.0)


class TestInterpolation:
    def test_endpoints_exact(self):
        a, b = random_se3("w", "x"), random_se3("w", "x")
        assert np.allclose(interpolate(a, b, 0.0).as_matrix(), a.as_matrix(), atol=1e-9)
        assert np.allclose(interpolate(a, b, 1.0).as_matrix(), b.as_matrix(), atol=1e-9)

    def test_translation_is_straight_line(self):
        """Deliberately NOT a screw motion -- see interpolate()'s docstring."""
        a, b = random_se3(), random_se3()
        mid = interpolate(a, b, 0.25)
        assert np.allclose(mid.t, 0.75 * a.t + 0.25 * b.t, atol=1e-9)

    def test_takes_short_way_round(self):
        """A 350-degree detour is the classic quaternion double-cover bug."""
        a = SE3.from_euler("z", [np.deg2rad(-170)])
        b = SE3.from_euler("z", [np.deg2rad(170)])
        # Shorter arc is 20 deg through +/-180, not 340 deg through zero.
        steps = [interpolate(a, b, s) for s in np.linspace(0, 1, 40)]
        total = sum(steps[i].distance_to(steps[i + 1])[1] for i in range(len(steps) - 1))
        assert np.degrees(total) < 30, f"took the long way: {np.degrees(total):.0f} deg"

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            interpolate(random_se3(), random_se3(), 1.5)


class TestMetrics:
    def test_distance_to_self_is_zero(self):
        T = random_se3()
        d, a = T.distance_to(T)
        assert d < 1e-12 and a < 1e-12

    def test_distance_is_pure_translation_when_rotation_equal(self):
        Rm = R.random(random_state=3).as_matrix()
        a = SE3.from_rt(Rm, [0, 0, 0])
        b = SE3.from_rt(Rm, [0.3, 0.4, 0.0])
        d, ang = a.distance_to(b)
        assert np.isclose(d, 0.5) and ang < 1e-12
