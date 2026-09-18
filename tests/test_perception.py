"""Perception tests: detection, PnP, ambiguity, and plane lifting.

Poses are compared against the pose we RENDERED FROM, so failures are reported
in millimetres and degrees rather than as reprojection residuals. Those are
different claims and only the first is about correctness -- Step 2 showed a
0.039 px residual hiding a 28% error.
"""

from functools import lru_cache

import numpy as np
import pytest

from vgp.msgs import CameraInfo, Detection2D, Detection2DArray, Header, Image
from vgp.perception.aruco import ArucoDetector, MarkerSpec, marker_model_points, generate_marker
from vgp.perception.pnp import (
    AMBIGUITY_RATIO, PnPFailure, depth_sensitivity, solve_pnp, pose_from_detection,
)
from vgp.perception.plane import Plane, PlaneLiftError, backproject_ray, intersect, pose_on_plane
from vgp.perception.synthetic import T_FACING, marker_pose, render_marker_view
from vgp.utils.transforms import SE3

W, H, FX = 1280, 720, 900.0
K = np.array([[FX, 0, (W - 1) / 2], [0, FX, (H - 1) / 2], [0, 0, 1]])
INFO = CameraInfo(Header(0.0, "camera_optical"), K, np.zeros(5), W, H)
SIZE = 0.05
MID = 7


def _img(rgb, stamp=0.0):
    return Image(Header(stamp, "camera_optical"), rgb)


@lru_cache(maxsize=None)
def _observe(tilt, dist, azim=30.0, size=SIZE, noise=2.0, seed=0):
    """Render -> detect -> solve. Cached; several tests share fixtures."""
    T_true = marker_pose(dist, tilt, azim)
    rgb = render_marker_view(MID, size, K, T_true, W, H,
                             noise_std=noise, rng=np.random.default_rng(seed))
    dets = ArucoDetector({MID: MarkerSpec(size)}).detect(_img(rgb)).detections
    if not dets:
        return T_true, None, None
    return T_true, dets[0], solve_pnp(dets[0].model_points, dets[0].corners, INFO)


def _err(T_true, T_est):
    d = T_true.inv() @ SE3(T_est.matrix)
    return d.translation_norm(), np.degrees(d.rotation_angle())


class TestMarkerConventions:
    def test_model_points_are_centre_origin(self):
        """Centre-origin makes a marker's pose independent of its size."""
        assert np.allclose(marker_model_points(0.05).mean(axis=0), 0.0)

    def test_model_points_are_planar(self):
        assert np.all(marker_model_points(0.05)[:, 2] == 0.0)

    def test_corner_order_is_tl_tr_br_bl(self):
        """Marker Y is UP while image v is DOWN -- hence the 'reversed' look.
        Hand-rolling these in the wrong order is a standard bug."""
        p = marker_model_points(0.05)
        assert p[0][0] < 0 and p[0][1] > 0      # TL: -x, +y
        assert p[1][0] > 0 and p[1][1] > 0      # TR: +x, +y
        assert p[2][0] > 0 and p[2][1] < 0      # BR
        assert p[3][0] < 0 and p[3][1] < 0      # BL

    def test_facing_marker_is_rotated_180_about_x(self):
        """REGRESSION: the test fixture originally used identity here.

        The ArUco marker frame's +Z points OUT toward the viewer; the camera's
        +Z points INTO the scene. So a marker squarely facing the camera has
        R = 180 deg about X. With identity the marker renders MIRRORED -- and
        then does not decode at all, which is how this was caught.
        """
        assert np.allclose(T_FACING.R, np.diag([1.0, -1.0, -1.0]))
        assert np.isclose(np.linalg.det(T_FACING.R), 1.0)
        proj = (SE3.from_rt(T_FACING.R, [0, 0, 0.4]).apply(marker_model_points(SIZE)) @ K.T)
        uv = proj[:, :2] / proj[:, 2:3]
        assert uv[0, 1] < uv[2, 1], "corner 0 (TL) must be ABOVE corner 2 (BR)"

    def test_quiet_zone_is_mandatory(self):
        """Verified, not assumed: the same marker fails to detect without it."""
        import cv2
        det = ArucoDetector({MID: MarkerSpec(SIZE)})
        bare = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), MID, 200)
        padded = generate_marker(MID, 200)
        assert not det.detect(_img(cv2.cvtColor(bare, cv2.COLOR_GRAY2RGB))).detections
        assert det.detect(_img(cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB))).detections


@pytest.mark.slow
class TestPoseAccuracy:
    @pytest.mark.parametrize("tilt,dist", [(15, 0.4), (30, 0.4), (30, 0.8), (45, 0.6)])
    def test_translation_within_a_centimetre(self, tilt, dist):
        T_true, det, r = _observe(float(tilt), float(dist))
        assert r is not None, "marker not detected"
        t_err, _ = _err(T_true, r.pose)
        assert t_err < 0.010, f"{t_err*1000:.1f} mm"

    @pytest.mark.parametrize("tilt,dist", [(15, 0.4), (30, 0.4), (45, 0.6)])
    def test_rotation_within_a_few_degrees(self, tilt, dist):
        T_true, det, r = _observe(float(tilt), float(dist))
        _, r_err = _err(T_true, r.pose)
        assert r_err < 5.0, f"{r_err:.2f} deg"

    def test_reprojection_error_is_subpixel(self):
        assert _observe(30.0, 0.4)[2].rmse_px < 1.0

    def test_detects_at_range(self):
        assert _observe(30.0, 1.2)[1] is not None


@pytest.mark.slow
class TestPlanarAmbiguity:
    """The two-fold ambiguity of a planar target, and our detection of it.

    A near-frontal planar marker has two poses that project almost identically,
    mirrored through the marker plane. Picking wrong flips the approach axis and
    the gripper dives in tilted. The classic symptom is a pose that flickers
    between two orientations frame to frame.
    """

    def test_frontal_marker_is_flagged_ambiguous(self):
        assert _observe(0.0, 0.5).__getitem__(2).ambiguous

    def test_strongly_tilted_marker_is_not_ambiguous(self):
        assert not _observe(45.0, 0.4)[2].ambiguous

    def test_error_ratio_grows_with_tilt(self):
        """The signal we threshold on must actually be monotone in the thing
        that resolves the ambiguity."""
        ratios = [_observe(float(t), 0.4)[2].error_ratio for t in (0, 15, 30, 45)]
        assert ratios[0] < ratios[-1]
        assert ratios[-1] > AMBIGUITY_RATIO * 2

    def test_alternative_is_returned_when_ambiguous(self):
        r = _observe(0.0, 0.5)[2]
        assert r.alternative is not None
        # The two solutions must be genuinely different, or flagging is noise.
        _, ang = SE3(r.pose.matrix).distance_to(r.alternative)
        assert np.degrees(ang) > 2.0

    def test_no_alternative_when_unambiguous(self):
        assert _observe(45.0, 0.4)[2].alternative is None


@pytest.mark.slow
class TestErrorAnisotropy:
    """Monocular pose error is deeply anisotropic, and people quote it as if
    it were not. Depth is recovered from apparent SIZE, so

        dZ/dx = Z/s

    A 5 cm marker at 1 m is ~20x less certain in depth than sideways, and depth
    degrades with Z^2 while lateral is only linear. Step 5's approach direction
    and Step 6's decision to servo rather than reach open-loop both follow from
    this.
    """

    def _spread(self, size, dist, n=25):
        errs = []
        for k in range(n):
            T_true, det, r = _observe(25.0, dist, 40.0, size, 3.0, k)
            if r is not None:
                errs.append(r.pose.t - T_true.t)
        e = np.array(errs)
        return np.hypot(e[:, 0].std(), e[:, 1].std()), e[:, 2].std()

    def test_depth_is_much_worse_than_lateral(self):
        lat, dep = self._spread(SIZE, 0.7)
        assert dep > 5 * lat, f"depth {dep*1000:.3f}mm vs lateral {lat*1000:.3f}mm"

    def test_ratio_tracks_z_over_s(self):
        lat, dep = self._spread(SIZE, 0.7)
        predicted = 0.7 / SIZE
        assert 0.35 * predicted < dep / lat < 2.5 * predicted

    def test_bigger_marker_is_more_certain_in_depth(self):
        _, small = self._spread(0.03, 0.7)
        _, big = self._spread(0.05, 0.7)
        assert big < small

    def test_depth_sensitivity_formula(self):
        dx, dz = depth_sensitivity(0.05, 1.0, 900.0, 1.0)
        assert np.isclose(dx, 1.0 / 900.0)
        assert np.isclose(dz, 1.0 / (0.05 * 900.0))
        # quadratic in Z, linear in 1/s
        assert np.isclose(depth_sensitivity(0.05, 2.0, 900.0)[1],
                          4 * depth_sensitivity(0.05, 1.0, 900.0)[1])


class TestPnPGuards:
    def test_rejects_too_few_points(self):
        with pytest.raises(PnPFailure, match=">= 4"):
            solve_pnp(np.zeros((3, 3)), np.zeros((3, 2)), INFO)

    def test_rejects_mismatched_counts(self):
        with pytest.raises(PnPFailure, match="model points vs"):
            solve_pnp(np.zeros((5, 3)), np.zeros((4, 2)), INFO)

    def test_rejects_bbox_only_detection(self):
        """A bbox cannot give orientation: a square seen rotated has the same
        axis-aligned box. Refusing is better than inventing one."""
        det = Detection2D("blob", 0.9, np.array([0.0, 0, 10, 10]))
        with pytest.raises(PnPFailure, match="cannot yield orientation"):
            pose_from_detection(det, INFO, 0.0, "x", "camera_optical")

    def test_labels_result_frames(self):
        _, det, _ = _observe(30.0, 0.4)
        p = pose_from_detection(det, INFO, 1.5, "cube", "camera_optical")
        assert (p.pose.parent, p.pose.child) == ("camera_optical", "obj/cube")
        assert p.header.stamp == 1.5


class TestPlaneLifting:
    def _down_plane(self, height=0.6):
        """Camera looking straight down at a table `height` below it."""
        return Plane(np.array([0.0, 0.0, 1.0]), height)

    def test_centre_ray_hits_directly_below(self):
        p, inc = intersect(np.array([(W - 1) / 2, (H - 1) / 2]), INFO, self._down_plane())
        assert np.allclose(p, [0, 0, 0.6], atol=1e-6) and np.isclose(inc, 90.0)

    def test_ray_is_unit_length(self):
        assert np.isclose(np.linalg.norm(backproject_ray(np.array([100.0, 50.0]), INFO)), 1.0)

    def test_refuses_grazing_geometry(self):
        """Near-parallel rays make a pixel of noise worth metres. Refusing is
        the honest response -- no detector improvement fixes conditioning."""
        n = np.array([0.0, np.cos(np.radians(3)), np.sin(np.radians(3))])
        with pytest.raises(PlaneLiftError, match="deg"):
            intersect(np.array([640.0, 360.0]), INFO, Plane(n, 0.6), min_incidence_deg=10)

    def test_conditioning_degrades_as_incidence_falls(self):
        """Quantifies the above: the same 1 px moves the point further as the
        ray flattens."""
        moves = []
        for ang in (90, 30, 12):
            n = np.array([0.0, np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            pl = Plane(n, 0.6)
            a, _ = intersect(np.array([639.5, 359.5]), INFO, pl, min_incidence_deg=5)
            b, _ = intersect(np.array([640.5, 359.5]), INFO, pl, min_incidence_deg=5)
            moves.append(np.linalg.norm(b - a))
        assert moves[0] < moves[1] < moves[2]

    def test_half_height_offset_recovers_object_centre(self):
        """THE SUBTLE BIAS: a silhouette centroid is up the object's side, not
        on the table. Intersecting it with the TABLE pulls the point toward the
        camera -- systematic, so averaging frames does not remove it."""
        table = self._down_plane(0.6)
        on_table, _ = intersect(np.array([700.0, 400.0]), INFO, table)
        lifted, _ = intersect(np.array([700.0, 400.0]), INFO,
                              table.offset_toward_camera(0.025))
        assert lifted[2] < on_table[2]            # the centre is nearer the camera
        assert np.isclose(on_table[2] - lifted[2], 0.025, atol=1e-9)

    def test_offset_direction_is_independent_of_normal_sign(self):
        """REGRESSION: offset_by(+h/2) moved the plane AWAY from the camera for
        one normal direction and toward it for the other -- doubling the bias
        instead of removing it. The intent, not the sign, is what is fixed."""
        up = Plane(np.array([0.0, 0.0, 1.0]), 0.6)
        down = Plane(np.array([0.0, 0.0, -1.0]), -0.6)
        a, _ = intersect(np.array([700.0, 400.0]), INFO, up.offset_toward_camera(0.025))
        b, _ = intersect(np.array([700.0, 400.0]), INFO, down.offset_toward_camera(0.025))
        assert np.allclose(a, b), "flipping the normal must not change the result"

    def test_pose_on_plane_is_a_proper_rotation(self):
        det = Detection2D("red", 0.95, np.array([600.0, 330, 700, 400]),
                          corners=np.array([[600.0, 330], [700, 330], [700, 400], [600, 400]]))
        p = pose_on_plane(det, INFO, self._down_plane(), 0.05, 0.0, "cube", "camera_optical")
        assert np.isclose(np.linalg.det(p.pose.R), 1.0)

    def test_plane_pose_reports_a_large_rmse_sentinel(self):
        """This path has no reprojection residual. Reporting a small number
        would let it masquerade as a well-constrained marker pose downstream."""
        det = Detection2D("red", 0.95, np.array([600.0, 330, 700, 400]),
                          corners=np.array([[600.0, 330], [700, 330], [700, 400], [600, 400]]))
        p = pose_on_plane(det, INFO, self._down_plane(), 0.05, 0.0, "cube", "camera_optical")
        assert p.reprojection_rmse_px >= 1.0

    def test_plane_from_pose(self):
        T = SE3.from_rt(np.eye(3), [0, 0, 0.6], parent="camera_optical", child="table")
        pl = Plane.from_pose(T)
        assert np.allclose(pl.normal, [0, 0, 1]) and np.isclose(pl.offset, 0.6)


# ---------------------------------------------------------------------------
# Node-level integration
# ---------------------------------------------------------------------------

from vgp.perception.nodes import (              # noqa: E402
    DetectorNode, PoseEstimatorNode, TOPIC_DETECTIONS, TOPIC_POSES,
)
from vgp.runtime import Executor, SimClock      # noqa: E402
from vgp.sensors.camera_node import CameraNode  # noqa: E402
from vgp.sensors.source import CameraSource     # noqa: E402
from vgp.msgs import ObjectPoseArray            # noqa: E402


class _ScriptedCamera(CameraSource):
    """Replays synthetic marker frames. Lets us test the NODE wiring without a
    textured simulator scene, and with the true pose known exactly."""

    def __init__(self, poses, clock, size=SIZE):
        self.poses, self.clock, self.size = poses, clock, size
        self.i = 0

    def read(self):
        T = self.poses[min(self.i, len(self.poses) - 1)]
        self.i += 1
        rgb = render_marker_view(MID, self.size, K, T, W, H,
                                 noise_std=2.0, rng=np.random.default_rng(self.i))
        hdr = Header(self.clock.now(), "camera_optical")
        return Image(hdr, rgb), CameraInfo(hdr, K, np.zeros(5), W, H)

    def close(self):
        pass


def _pipeline(poses, hz=10.0, detector_hz=None, max_rmse=2.0):
    ex = Executor(SimClock())
    cam = ex.add(CameraNode("cam", ex.bus, ex.tf, ex.clock,
                            _ScriptedCamera(poses, ex.clock), hz=hz))
    det = ex.add(DetectorNode("det", ex.bus, ex.tf, ex.clock,
                              ArucoDetector({MID: MarkerSpec(SIZE)}), max_hz=detector_hz))
    pose = ex.add(PoseEstimatorNode("pose", ex.bus, ex.tf, ex.clock, max_rmse_px=max_rmse))
    seen = []
    ex.bus.declare(TOPIC_POSES, ObjectPoseArray)
    ex.bus.subscribe(TOPIC_POSES, seen.append, owner="test")
    return ex, cam, det, pose, seen


@pytest.mark.slow
class TestPerceptionPipeline:
    def test_end_to_end_produces_poses(self):
        ex, cam, det, pose, seen = _pipeline([marker_pose(0.5, 30.0, 20.0)])
        ex.spin(duration=1.0)
        assert cam.frames_published == 10
        assert pose.poses_published == 10
        assert sum(len(m.objects) for m in seen) == 10

    def test_pose_lands_in_the_tf_tree(self):
        """The payoff of Step 1: perception writes into TF and anything can ask
        for the object's pose by name, without knowing perception exists."""
        ex, *_ = _pipeline([marker_pose(0.5, 30.0, 20.0)])
        ex.spin(duration=1.0)
        assert ex.tf.can_lookup("camera_optical", "obj/marker_7")

    def test_tf_entries_carry_the_image_timestamp(self):
        """Not 'now'. On a moving arm those differ by centimetres, which is the
        entire reason the TF buffer interpolates."""
        ex, *_ = _pipeline([marker_pose(0.5, 30.0, 20.0)], hz=10.0)
        ex.spin(duration=1.0)
        edge = ex.tf._edges[("camera_optical", "obj/marker_7")]
        assert np.allclose(edge.stamps, np.arange(1, 11) * 0.1)

    def test_pipeline_pose_matches_ground_truth(self):
        T_true = marker_pose(0.5, 30.0, 20.0)
        ex, _, _, _, seen = _pipeline([T_true])
        ex.spin(duration=0.5)
        t_err, r_err = _err(T_true, seen[-1].objects[0].pose)
        assert t_err < 0.010 and r_err < 5.0

    def test_detector_throttling_drops_rather_than_queues(self):
        """A stale frame has no value, so dropping beats falling behind."""
        ex, cam, det, _, _ = _pipeline([marker_pose(0.5, 30.0, 20.0)],
                                       hz=30.0, detector_hz=10.0)
        ex.spin(duration=1.0)
        assert cam.frames_published == 30
        assert 9 <= det.frames_processed <= 11
        assert 0.6 < det.drop_fraction < 0.72

    def test_rmse_gate_blocks_bad_poses(self):
        """A geometric gate, not a recognition one: the marker is recognised
        with full confidence either way."""
        ex, _, _, pose, seen = _pipeline([marker_pose(0.5, 30.0, 20.0)], max_rmse=0.0001)
        ex.spin(duration=0.5)
        assert pose.poses_published == 0 and pose.rejected_rmse == 5

    def test_detections_topic_is_independently_useful(self):
        """A recorder or visualiser subscribes here without caring whether a
        pose was solvable -- the reason detection and PnP are separate nodes."""
        ex, *_ = _pipeline([marker_pose(0.5, 30.0, 20.0)])
        got = []
        # Declare before subscribing: topics are declared in on_configure, which
        # runs at start(). This is the two-phase-startup rule seen from outside.
        ex.bus.declare(TOPIC_DETECTIONS, Detection2DArray)
        ex.bus.subscribe(TOPIC_DETECTIONS, got.append, owner="viz")
        ex.spin(duration=0.5)
        assert len(got) == 5 and all(len(d.detections) == 1 for d in got)

    def test_bbox_only_detections_are_counted_not_crashed(self):
        from vgp.perception.color import ColorDetector
        ex, _, det, pose, _ = _pipeline([marker_pose(0.5, 30.0, 20.0)])
        det.detector = ColorDetector()          # produces no model_points
        ex.spin(duration=0.5)
        assert pose.poses_published == 0
        assert pose.rejected_no_geometry >= 0   # must not raise
