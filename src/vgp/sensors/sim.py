"""MuJoCo camera source: rendered frames plus privileged ground truth.

THE CONVERSION THAT MATTERS HERE
--------------------------------
MuJoCo inherits OpenGL's camera convention; OpenCV uses its own. They differ,
and the difference is not a rounding error:

    MuJoCo/OpenGL camera:  looks down -Z,  +X right, +Y UP
    OpenCV optical:        looks down +Z,  +X right, +Y DOWN

So the mapping is a 180-degree rotation about X -- `diag(1, -1, -1)`.

Get this wrong and the symptom is beautifully misleading: the image renders
fine, PnP returns a pose with LOW reprojection error (the geometry is
self-consistent), and the arm reaches to a point mirrored through the camera
axis. Everything looks healthy except the robot. We verify it against actually
rendered pixels rather than trusting the reasoning.

Note `diag(1,-1,-1)` is its own inverse, which is why one constant serves both
directions -- pleasant, but do not let it lull you into skipping the direction
check when adapting this to a different simulator.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - dependency is declared, this aids the error message
    raise ImportError("MuJoCo is required for the sim camera: pip install mujoco")

from vgp.msgs import CameraInfo, Header, Image
from vgp.sensors.source import CameraSource, GroundTruth, intrinsics_from_fov
from vgp.utils.frames import CAMERA_OPTICAL
from vgp.utils.transforms import SE3

# Maps a point in MuJoCo/OpenGL camera coords into OpenCV optical coords.
# Self-inverse, so the same constant works both ways.
T_OPTICAL_FROM_MUJOCO_CAM = SE3.from_rt(np.diag([1.0, -1.0, -1.0]), np.zeros(3))


class SimCameraSource(CameraSource):
    """Renders `camera_name` from a MuJoCo model.

    Does NOT own the model or step physics -- it only looks. Keeping rendering
    separate from simulation means the same source works whether physics is
    driven by a test, by the task state machine, or not at all (a static scene
    for calibration). It also means two cameras are two instances over one
    model, with no coordination needed.
    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        camera_name: str,
        width: int = 640,
        height: int = 480,
        frame_id: str = CAMERA_OPTICAL,
        noise_std: float = 0.0,
    ) -> None:
        self.model, self.data = model, data
        self.camera_name = camera_name
        self.width, self.height = width, height
        self.frame_id = frame_id
        # Optional sensor noise. Default 0 so tests are exact; turned UP in the
        # Step 7 eval to measure how gracefully the pose estimate degrades.
        # A pipeline only ever tested on noiseless renders is a pipeline whose
        # thresholds are guesses.
        self.noise_std = noise_std
        self._rng = np.random.default_rng(0)

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(model.ncam)
            ]
            raise ValueError(f"no camera '{camera_name}' in model; have {names}")

        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._info = intrinsics_from_fov(
            width, height, float(model.cam_fovy[self.cam_id]), frame_id=frame_id
        )

    # ------------------------------------------------------------------ read

    def read(self) -> tuple[Image, CameraInfo]:
        stamp = float(self.data.time)
        self._renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self._renderer.render()                      # MuJoCo renders RGB
        if self.noise_std > 0:
            noisy = rgb.astype(np.float32) + self._rng.normal(0, self.noise_std, rgb.shape)
            rgb = np.clip(noisy, 0, 255).astype(np.uint8)
        img = Image(Header(stamp, self.frame_id), np.ascontiguousarray(rgb), "rgb8")
        info = CameraInfo(
            Header(stamp, self.frame_id), self._info.K, self._info.dist,
            self.width, self.height,
        )
        return img, info

    def close(self) -> None:
        self._renderer.close()

    # ---------------------------------------------------------- ground truth

    def camera_pose(self) -> SE3:
        """T_world_cameraOptical, exact.

        `cam_xmat` is row-major 3x3 in world coordinates, describing the MuJoCo
        camera frame. We post-multiply by the OpenGL->OpenCV constant to land in
        the optical convention everything downstream expects.
        """
        R_world_mjcam = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        t = self.data.cam_xpos[self.cam_id]
        T_world_mjcam = SE3.from_rt(R_world_mjcam, t, parent="world", child="mjcam")
        return SE3(
            (T_world_mjcam @ SE3(T_OPTICAL_FROM_MUJOCO_CAM.matrix, "mjcam", "opt")).matrix,
            parent="world", child=self.frame_id,
        )

    def body_pose(self, body_name: str) -> SE3:
        """Exact T_world_body for any body -- the reference for pose error."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"no body '{body_name}' in model")
        return SE3.from_quat_wxyz(         # MuJoCo stores (w,x,y,z) -- hence this ctor
            self.data.xquat[bid], self.data.xpos[bid], parent="world", child=body_name,
        )

    def ground_truth(self, body_names: tuple[str, ...] = ()) -> GroundTruth:
        return GroundTruth(
            stamp=float(self.data.time),
            camera_pose=self.camera_pose(),
            object_poses={n: self.body_pose(n) for n in body_names},
        )

    # -------------------------------------------------------------- utility

    def project(self, p_world: np.ndarray) -> np.ndarray:
        """Project world points to pixels using our own K and camera pose.

        This is the oracle the tests compare against the RENDERER's own output.
        If our intrinsics or the OpenGL->OpenCV rotation were wrong, these two
        would disagree -- which is exactly the check that pins both down.
        """
        p_opt = self.camera_pose().inv().apply(np.atleast_2d(p_world))
        if np.any(p_opt[:, 2] <= 0):
            raise ValueError("point is behind the camera (optical Z <= 0)")
        uvw = p_opt @ self._info.K.T
        return uvw[:, :2] / uvw[:, 2:3]
