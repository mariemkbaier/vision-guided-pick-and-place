"""Calibration tests against a KNOWN ground-truth K.

The point of synthesising views rather than using real photos is that we can
assert on the ANSWER, not just the residual. "RMSE was 0.03 px" is not a claim
about correctness -- see TestFrontoParallelDegeneracy, where the residual is
excellent and the focal length is ~15% wrong.
"""

from functools import lru_cache

import numpy as np
import pytest

from vgp.calib.intrinsics import (
    CheckerboardSpec, IntrinsicCalibrator, recommend_board,
    save_intrinsics, load_intrinsics,
)
from vgp.calib.synthetic import render_view, pose_grid, board_image

W, H = 960, 540
FX = 780.0
K_TRUE = np.array([[FX, 0, (W - 1) / 2], [0, FX, (H - 1) / 2], [0, 0, 1]])
DIST_TRUE = np.array([-0.15, 0.05, 0.001, -0.001, 0.0])

# Aspect-matched to the sensor. A 9x6 board on a 16:9 sensor CANNOT reach the
# coverage gate no matter how well you wave it around -- see recommend_board.
SPEC = recommend_board(W, H, 0.02)


@lru_cache(maxsize=None)
def _result(n=20, tilt=35.0, lateral=0.30, with_dist=True, seed=0, noise=3.0):
    """Cached: detection dominates runtime and several tests share fixtures."""
    cal = IntrinsicCalibrator(SPEC)
    dist = DIST_TRUE if with_dist else None
    for p in pose_grid(SPEC, n, K_TRUE, W, H, max_tilt_deg=tilt, lateral=lateral, seed=seed):
        try:
            cal.add_view(render_view(SPEC, K_TRUE, p, W, H, dist=dist,
                                     noise_std=noise, rng=np.random.default_rng(seed)))
        except Exception:
            pass
    return cal.calibrate()


class TestBoardGeometry:
    def test_inner_corner_convention(self):
        """cols x rows INNER corners = (cols+1) x (rows+1) squares."""
        s = CheckerboardSpec(9, 6, 0.025)
        assert s.object_points().shape == (54, 3)

    def test_points_are_planar(self):
        assert np.all(SPEC.object_points()[:, 2] == 0.0)

    def test_spacing_is_square_size(self):
        p = SPEC.object_points().reshape(SPEC.rows, SPEC.cols, 3)
        assert np.isclose(np.linalg.norm(p[0, 1] - p[0, 0]), SPEC.square_size_m)

    def test_board_image_renders(self):
        img, M = board_image(SPEC)
        assert img.ndim == 2 and img.min() == 0 and img.max() == 255
        assert M.shape == (3, 3)


class TestBoardRecommendation:
    """Aspect mismatch is a property of the TARGET, not the operator."""

    @pytest.mark.parametrize("w,h", [(1280, 720), (640, 480), (1920, 1080), (1600, 1200)])
    def test_matches_sensor_aspect(self, w, h):
        b = recommend_board(w, h)
        assert abs((b.cols + 2) / (b.rows + 2) - w / h) < 0.08

    def test_classic_9x6_is_wrong_for_16x9(self):
        """The default board everyone uses is 3:2 -- a permanent handicap on a
        widescreen sensor, unfixable by better technique."""
        assert recommend_board(1920, 1080) != CheckerboardSpec(9, 6)


@pytest.mark.slow
class TestRecoversKnownIntrinsics:
    def test_recovers_focal_length(self):
        assert abs(_result().camera_info.fx - FX) / FX < 0.01

    def test_recovers_principal_point(self):
        ci = _result().camera_info
        assert abs(ci.cx - K_TRUE[0, 2]) < 5.0 and abs(ci.cy - K_TRUE[1, 2]) < 5.0

    def test_recovers_radial_distortion(self):
        """Regression guard: applying distortion with the REMOVAL map (the
        natural mistake, since OpenCV only ships that direction) recovers
        k1 = +0.15 for a true -0.15 -- sign-flipped, not merely noisy."""
        k1 = _result().camera_info.dist[0]
        assert abs(k1 - DIST_TRUE[0]) < 0.03, f"k1={k1:+.3f}, expected {DIST_TRUE[0]:+.3f}"
        assert np.sign(k1) == np.sign(DIST_TRUE[0]), "sign flip = inverted distortion map"

    def test_reprojection_error_is_subpixel(self):
        assert _result().rms < 1.0

    def test_per_view_errors_reported(self):
        r = _result()
        assert len(r.per_view_rms) == r.n_views

    def test_all_gates_pass_on_good_data(self):
        r = _result()
        assert r.ok, f"good data should pass all gates, got: {r.problems()}"


@pytest.mark.slow
class TestFrontoParallelDegeneracy:
    """The headline result of Step 2.

    Face-on views leave focal length and distance nearly unidentifiable: a board
    twice as far with twice the focal length makes almost the same image. The
    optimiser lands somewhere along that valley and reports an EXCELLENT
    residual while fx is badly wrong.

    Measured here: the face-on set achieves a reprojection RMSE *as good as or
    better than* the well-conditioned set, with a focal length ~15% wrong. That
    is precisely why the gate is on the DATA, not on the residual.
    """

    def test_focal_length_wrong_despite_excellent_rmse(self):
        bad = _result(tilt=0.0)
        err = abs(bad.camera_info.fx - FX) / FX
        assert bad.rms < 0.5, "residual should look fine -- that IS the trap"
        assert err > 0.05, f"expected a badly wrong fx, got {err:.2%}"

    def test_bad_residual_is_not_worse_than_good_one(self):
        """The counter-intuitive part, asserted so it cannot quietly stop being
        true: you genuinely cannot tell these apart by residual."""
        assert _result(tilt=0.0).rms <= _result(tilt=35.0).rms * 1.5

    def test_tilt_gate_catches_it(self):
        assert any("tilt" in p for p in _result(tilt=0.0).problems())

    def test_tilted_views_pass_the_tilt_gate(self):
        assert not any("tilt" in p for p in _result(tilt=35.0).problems())


@pytest.mark.slow
class TestQualityGates:
    def test_central_views_flagged_for_coverage(self):
        assert any("occupy only" in p for p in _result(lateral=0.02).problems())

    def test_too_few_views_flagged(self):
        assert any("views" in p for p in _result(n=6).problems())

    def test_refuses_fewer_than_three_views(self):
        with pytest.raises(ValueError, match="at least 3"):
            IntrinsicCalibrator(SPEC).calibrate()

    def test_rejects_mixed_resolutions(self):
        """Silently averaging two resolutions' K would be a large focal error."""
        cal = IntrinsicCalibrator(SPEC)
        p = pose_grid(SPEC, 1, K_TRUE, W, H, seed=0)[0]
        cal.add_view(render_view(SPEC, K_TRUE, p, W, H))
        with pytest.raises(ValueError, match="earlier views were"):
            cal.add_view(render_view(SPEC, K_TRUE, p, W // 2, H // 2))


@pytest.mark.slow
class TestVisibilitySampling:
    def test_sampled_poses_are_fully_visible(self):
        """A partly-visible checkerboard is detected NOT AT ALL, so naive
        lateral jitter silently discards ~45% of views -- and keeps the central,
        low-coverage ones."""
        poses = pose_grid(SPEC, 25, K_TRUE, W, H, max_tilt_deg=35.0, seed=2)
        pts = SPEC.object_points().astype(np.float64)
        for p in poses:
            q = p.apply(pts)
            uv = q @ K_TRUE.T
            uv = uv[:, :2] / uv[:, 2:3]
            assert uv[:, 0].min() >= 0 and uv[:, 0].max() < W
            assert uv[:, 1].min() >= 0 and uv[:, 1].max() < H

    def test_detection_rate_is_high(self):
        """If this drops, the fixture is throwing away views again."""
        poses = pose_grid(SPEC, 20, K_TRUE, W, H, max_tilt_deg=35.0, seed=0)
        cal = IntrinsicCalibrator(SPEC)
        found = sum(cal.add_view(render_view(SPEC, K_TRUE, p, W, H, dist=DIST_TRUE))
                    for p in poses)
        assert found >= 18, f"only {found}/20 boards detected"


@pytest.mark.slow
class TestPersistence:
    def test_round_trip(self, tmp_path):
        r = _result()
        f = tmp_path / "cal.yaml"
        save_intrinsics(f, r)
        back = load_intrinsics(f)
        assert np.allclose(back.K, r.camera_info.K)
        assert np.allclose(back.dist, r.camera_info.dist)
        assert (back.width, back.height) == (W, H)

    def test_quality_metrics_persisted(self, tmp_path):
        """Six months later you must be able to judge the file without
        recalibrating the camera."""
        import yaml
        f = tmp_path / "cal.yaml"
        save_intrinsics(f, _result())
        q = yaml.safe_load(f.read_text())["quality"]
        assert {"reprojection_rms_px", "n_views", "tilt_span_deg", "problems"} <= set(q)

    def test_human_readable(self, tmp_path):
        f = tmp_path / "cal.yaml"
        save_intrinsics(f, _result())
        assert "camera_matrix" in f.read_text()


class TestCoverageMetric:
    """The metric itself needs testing: the first version was subtly wrong.

    It measured the FRACTION OF CORNERS falling in the outer margin. Measured
    from 15 to 40 views, hull coverage rose 52% -> 58% while that edge fraction
    FELL 4.0% -> 2.3%, because each extra view adds mostly central corners and
    dilutes the ratio. A quality metric that punishes collecting more data
    would have coached operators to stop early.
    """

    def test_occupancy_is_monotone_in_data(self):
        from vgp.calib.intrinsics import grid_occupancy
        rng = np.random.default_rng(0)
        pts = rng.uniform([0, 0], [W, H], size=(40, 2))
        prev = (0.0, 0.0)
        for n in (5, 10, 20, 40):
            got = grid_occupancy(pts[:n], W, H)
            assert got[0] >= prev[0] and got[1] >= prev[1], "adding views must never reduce coverage"
            prev = got

    def test_empty_input(self):
        from vgp.calib.intrinsics import grid_occupancy
        assert grid_occupancy(np.zeros((0, 2)), W, H) == (0.0, 0.0)

    def test_full_grid_scores_one(self):
        from vgp.calib.intrinsics import grid_occupancy, GRID_COLS, GRID_ROWS
        xs = (np.arange(GRID_COLS) + 0.5) / GRID_COLS * W
        ys = (np.arange(GRID_ROWS) + 0.5) / GRID_ROWS * H
        pts = np.array([[x, y] for x in xs for y in ys])
        assert grid_occupancy(pts, W, H) == (1.0, 1.0)

    def test_central_cluster_scores_low_on_border(self):
        from vgp.calib.intrinsics import grid_occupancy
        rng = np.random.default_rng(1)
        pts = rng.normal([W / 2, H / 2], [W / 30, H / 30], size=(500, 2))
        cov, edge = grid_occupancy(pts, W, H)
        assert edge < 0.1 and cov < 0.3

    def test_points_outside_image_are_clamped_not_crashing(self):
        from vgp.calib.intrinsics import grid_occupancy
        grid_occupancy(np.array([[-50.0, -50.0], [W + 99.0, H + 99.0]]), W, H)


class TestIntrinsicScaling:
    def test_all_four_parameters_scale(self):
        """Scaling fx/fy but not cx/cy is the classic half-done resize: it moves
        the principal point off centre and biases pose with image radius."""
        from vgp.msgs import CameraInfo, Header
        ci = CameraInfo(Header(0, "c"), K_TRUE.copy(), np.zeros(5), W, H)
        s = ci.scaled(0.5, 0.5)
        assert np.isclose(s.fx, FX * 0.5)
        assert np.isclose(s.cx, K_TRUE[0, 2] * 0.5)
        assert np.isclose(s.cy, K_TRUE[1, 2] * 0.5)

    def test_distortion_does_not_scale(self):
        """Distortion coefficients are dimensionless in normalised coords."""
        from vgp.msgs import CameraInfo, Header
        ci = CameraInfo(Header(0, "c"), K_TRUE.copy(), DIST_TRUE.copy(), W, H)
        assert np.allclose(ci.scaled(0.5, 0.5).dist, DIST_TRUE)
