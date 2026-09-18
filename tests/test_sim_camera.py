"""Sim camera tests: intrinsics and the OpenGL->OpenCV convention.

These check our model against ACTUAL RENDERED PIXELS rather than against the
derivation that produced it. That distinction is the point: a sign error in the
camera convention is perfectly self-consistent -- PnP would still report low
reprojection error -- so only real pixels can arbitrate.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vgp.sensors.sim import SimCameraSource, T_OPTICAL_FROM_MUJOCO_CAM
from vgp.sensors.source import intrinsics_from_fov

W, H, FOVY = 640, 480, 45.0

# Tiny bright spheres at exactly known world positions. Camera at (0,0,1) with
# its frame aligned to world, so it looks down world -Z (MuJoCo convention).
MARKERS = {
    "m_axis": (0.0, 0.0, 0.0),      # on the optical axis -> must land at (cx, cy)
    "m_px": (0.10, 0.0, 0.0),       # world +X -> image right
    "m_py": (0.0, 0.10, 0.0),       # world +Y -> image UP (smaller row)
    "m_far": (-0.15, 0.08, 0.0),
}


def _scene():
    bodies = "\n".join(
        f'<body name="{n}" pos="{x} {y} {z}"><geom type="sphere" size="0.004" '
        f'rgba="1 0 0 1" contype="0" conaffinity="0"/></body>'
        for n, (x, y, z) in MARKERS.items()
    )
    xml = f"""
    <mujoco>
      <visual><global offwidth="1280" offheight="960"/></visual>
      <asset><texture name="sky" type="skybox" builtin="flat" rgb1="0 0 0"
                      width="8" height="8"/></asset>
      <worldbody>
        <light pos="0 0 2" diffuse="1 1 1"/>
        {bodies}
        <camera name="cam" pos="0 0 1" xyaxes="1 0 0  0 1 0" fovy="{FOVY}"/>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _centroid(rgb, target=(255, 0, 0)):
    """Intensity-weighted centroid -- sub-pixel, which integer argmax is not."""
    f = rgb.astype(np.float32)
    w = np.clip(1.0 - np.linalg.norm(f - np.array(target, np.float32), axis=-1) / 140.0, 0, 1) ** 2
    if w.sum() < 1e-6:
        return None
    ys, xs = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    return np.array([(w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()])


def _render_only(src, model, data, keep):
    """Render with just one marker visible, so centroids never blend."""
    for n in MARKERS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        model.geom_rgba[model.body_geomadr[bid]] = [1, 0, 0, 1] if n == keep else [0, 0, 0, 0]
    mujoco.mj_forward(model, data)
    return src.read()[0].data


@pytest.fixture(scope="module")
def sim():
    m, d = _scene()
    src = SimCameraSource(m, d, "cam", W, H)
    yield src, m, d
    src.close()


@pytest.mark.slow
class TestIntrinsics:
    def test_principal_point_is_w_minus_1_over_2(self, sim):
        """THE half-pixel question, settled empirically.

        A marker on the optical axis lands exactly at the principal point. The
        two candidate conventions differ by 0.707 px diagonally -- small, but a
        SYSTEMATIC bias that PnP turns into a constant pose offset later
        misattributed to hand-eye calibration.
        """
        src, m, d = sim
        c = _centroid(_render_only(src, m, d, "m_axis"))
        assert np.allclose(c, [(W - 1) / 2, (H - 1) / 2], atol=0.05)
        assert not np.allclose(c, [W / 2, H / 2], atol=0.05), "W/2 would be wrong"

    def test_focal_length_matches_rendered_geometry(self, sim):
        """fy = (H/2)/tan(fovy/2), verified against where pixels actually land."""
        src, m, d = sim
        for name, p in MARKERS.items():
            got = _centroid(_render_only(src, m, d, name))
            want = src.project(np.array(p))[0]
            assert np.linalg.norm(got - want) < 0.25, f"{name}: {got} vs {want}"

    def test_fx_equals_fy_for_square_pixels(self, sim):
        info = sim[0]._info
        assert np.isclose(info.fx, info.fy)

    def test_zero_distortion(self, sim):
        """A render IS a perfect pinhole. Not true of the webcam -- which is a
        large part of why sim numbers flatter you."""
        assert np.allclose(sim[0]._info.dist, 0.0)

    @pytest.mark.parametrize("fovy", [30.0, 45.0, 60.0, 90.0])
    def test_fov_formula_over_range(self, fovy):
        info = intrinsics_from_fov(640, 480, fovy)
        half_height_at_1m = np.tan(np.deg2rad(fovy) / 2)
        assert np.isclose(info.fy, (480 / 2) / half_height_at_1m)


@pytest.mark.slow
class TestOpenGLToOpenCVConvention:
    def test_y_axis_flips(self, sim):
        """World +Y is UP in this camera, so it must render ABOVE centre
        (smaller row index). If the flip were missing the marker appears below
        and every pose is mirrored -- with a healthy-looking reprojection error.
        """
        src, m, d = sim
        c = _centroid(_render_only(src, m, d, "m_py"))
        assert c[1] < (H - 1) / 2 - 20, "world +Y must render above image centre"

    def test_x_axis_does_not_flip(self, sim):
        src, m, d = sim
        c = _centroid(_render_only(src, m, d, "m_px"))
        assert c[0] > (W - 1) / 2 + 20, "world +X must render right of centre"

    def test_conversion_is_self_inverse(self):
        T = T_OPTICAL_FROM_MUJOCO_CAM
        assert np.allclose((T @ T).as_matrix(), np.eye(4))

    def test_conversion_is_a_proper_rotation(self):
        """diag(1,-1,-1) has det +1. diag(1,1,-1) would be a REFLECTION and is
        the natural typo -- SE3's constructor rejects it, but assert it here
        too since this constant is copied between projects."""
        assert np.isclose(np.linalg.det(T_OPTICAL_FROM_MUJOCO_CAM.R), 1.0)

    def test_camera_pose_puts_scene_in_front(self, sim):
        """Optical Z must be POSITIVE for visible points. Negative means the
        camera convention is inverted and the scene is 'behind' the camera."""
        src, _, _ = sim
        for p in MARKERS.values():
            assert src.camera_pose().inv().apply(np.array(p))[2] > 0


@pytest.mark.slow
class TestGroundTruthIsolation:
    def test_read_returns_no_privileged_state(self, sim):
        """Ground truth must not be reachable from the normal message path, or
        every accuracy number downstream measures nothing."""
        img, info = sim[0].read()
        for msg in (img, info):
            assert not any("truth" in f or "pose" in f for f in vars(msg))

    def test_ground_truth_channel_gives_exact_poses(self, sim):
        src, _, _ = sim
        gt = src.ground_truth(tuple(MARKERS))
        for name, p in MARKERS.items():
            assert np.allclose(gt.object_poses[name].t, p, atol=1e-9)
