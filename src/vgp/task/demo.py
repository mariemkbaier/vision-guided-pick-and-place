"""Wire the whole pipeline together and run one pick-and-place.

This module is the answer to "show me the system": every component from Steps
1-6 appears exactly once, connected only through the bus and the TF tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    raise ImportError("MuJoCo required")

from vgp.calib.handeye import HandEyeCalibrator
from vgp.calib.handeye_collect import generate_poses
from vgp.msgs import Header, Image
from vgp.perception.aruco import ArucoDetector, MarkerSpec, select
from vgp.perception.nodes import DetectorNode, PoseEstimatorNode
from vgp.perception.pnp import solve_pnp
from vgp.robot.arm import Arm
from vgp.robot.node import RobotNode
from vgp.robot.scene import (
    HOME_Q, OBSERVE_Q, ObjectSpec, SceneSpec, build_xml, default_scene,
)
from vgp.runtime import Executor, SimClock
from vgp.sensors.camera_node import CameraNode
from vgp.sensors.sim import SimCameraSource
from vgp.task.pick_place import PickPlaceNode, State, TaskConfig
from vgp.utils.frames import T_CAMERA_LINK_FROM_OPTICAL
from vgp.utils.transforms import SE3


@dataclass
class DemoResult:
    success: bool
    final_state: str
    failure_reason: str
    handeye_error_mm: float
    handeye_error_deg: float
    servo_corrections_mm: list[float]
    object_start: np.ndarray
    object_end: np.ndarray
    place_error_mm: float
    sim_time: float
    log: str
    # Error attribution. `perception_error_mm` is the committed object-position
    # estimate against simulator truth, in base_link -- so it folds in PnP,
    # hand-eye and TF timing. Comparing runs with and without ground-truth
    # hand-eye separates those.
    perception_error_mm: float = float("nan")
    grasp_offset_mm: float = float("nan")


def true_hand_eye(arm: Arm, cam: SimCameraSource) -> SE3:
    """Ground-truth T_tool0_cameraOptical, read from the model.

    Available ONLY because this is a simulator, and used ONLY for grading --
    never fed into the pipeline. See sensors/source.py on why ground truth
    lives on its own channel.
    """
    return SE3((arm.fk_tool().inv() @ SE3(cam.camera_pose().matrix,
                                          "base_link", "camera_optical")).matrix,
               "tool0", "camera_optical")


def calibrate_hand_eye(model, data, arm, cam, spec, n=18, seed=1, noise_px=0.0):
    """Run the Step 4 calibration in sim and return (X_estimated, X_true)."""
    from scipy.spatial.transform import Rotation as R

    det = ArucoDetector({spec.calib_marker_id:
                         MarkerSpec(spec.calib_marker_size, "calib")})
    # Nominal guess used ONLY to generate viewpoints -- never in the solve.
    X_nom = SE3.from_rt(R.from_euler("xyz", [0, 0.30, 0.05]).as_matrix(),
                        [0.05, 0, -0.03], "tool0", "camera_optical")
    q_saved = arm.q.copy()
    qs = generate_poses(arm, spec.calib_marker_pos, X_nom, n=n, q_seed=HOME_Q, seed=seed)

    cal = HandEyeCalibrator()
    for q in qs:
        arm.q = q
        mujoco.mj_forward(model, data)
        img, info = cam.read()
        hit = select(det.detect(Image(Header(0, "camera_optical"), img.data)).detections,
                     "calib_0")
        if hit is None:
            continue
        r = solve_pnp(hit.model_points, hit.corners, info)
        if r.rmse_px > 1.5 or r.ambiguous:
            continue
        cal.add(arm.fk_tool(), SE3(r.pose.matrix, "camera_optical", "target"))

    X_true = true_hand_eye(arm, cam)
    res = cal.solve()
    arm.q = q_saved
    mujoco.mj_forward(model, data)
    return res, X_true


def make_scan_qs(arm: Arm, points, X_tool_cam_optical: SE3,
                 height: float = 0.40, tilt_deg: float = 20.0) -> tuple:
    """Joint configs whose CAMERA views each point from above.

    Multiple vantage points, because one is not enough. With objects placed
    anywhere on the table, a single fixed observation pose leaves part of the
    workspace outside the camera's 58-degree cone -- measured as "never saw a
    usable pose" on 25% of randomised trials, which reads as a perception
    failure and is actually a coverage failure.

    This is the eye-in-hand tax again (see scene.OBSERVE_Q): the camera goes
    where the arm puts it, so covering the workspace is a MOTION PLAN, not a
    property of the sensor.
    """
    from vgp.calib.handeye_collect import look_at
    from vgp.robot.ik import solve_ik_multistart
    from vgp.utils.frames import CAMERA_OPTICAL, TOOL

    # Take the tool->optical transform as an ARGUMENT rather than rebuilding it
    # from the model.
    #
    # The trap this avoids: our MJCF orients the camera_link BODY in the
    # optical convention (Z forward), so for this particular model
    # T_link_optical is identity -- while utils/frames exports the REP-103
    # constant, which is not. The TF chain stays correct either way because the
    # robot node divides the same constant back out, but composing them BY HAND
    # applies a rotation that is not there. Measured: 120 degrees of error,
    # position perfectly correct, cameras aimed at a point 1.2 m behind the
    # robot. Nothing raises; the camera simply sees nothing.
    T_tool_cam = SE3(X_tool_cam_optical.matrix)
    out = []
    for i, pt in enumerate(points):
        az = np.deg2rad(180.0 + 40.0 * i)
        tilt = np.deg2rad(tilt_deg)
        pt = np.array([pt[0], pt[1], 0.0]) if len(pt) == 2 else np.asarray(pt, float)
        eye = pt + height * np.array([np.sin(tilt) * np.cos(az),
                                      np.sin(tilt) * np.sin(az), np.cos(tilt)])
        T_base_cam = look_at(eye, pt)
        # Convert the desired CAMERA pose to a TOOL pose. Uses the nominal
        # mount, exactly like hand-eye pose generation: it only has to put the
        # workspace in frame, not be accurate.
        T_base_tool = SE3((T_base_cam @ SE3(T_tool_cam.matrix).inv()).matrix)
        r = solve_ik_multistart(arm, SE3(T_base_tool.matrix, "base_link", "tool0"),
                                site=arm.tool_sid, q_rest=HOME_Q,
                                pos_tol=2e-3, rot_tol=2e-2)
        if r.success:
            out.append(r.q.copy())
    return tuple(out)


@dataclass
class Pipeline:
    """Everything wired together, before it is run.

    Exposed as a separate step from `run_demo` so a VIEWER can drive the same
    pipeline the headless demo drives -- rather than a visualisation script
    re-wiring its own copy, which would then drift out of sync with what the
    tests actually exercise.
    """

    model: object
    data: object
    arm: Arm
    camera: SimCameraSource
    executor: Executor
    task: PickPlaceNode
    obj: ObjectSpec
    scene: SceneSpec
    handeye_error_mm: float
    handeye_error_deg: float
    object_start: np.ndarray


def build_pipeline(
    scene: SceneSpec | None = None,
    target: str = "cube_1",
    object_index: int | None = None,
    use_true_hand_eye: bool = False,
    camera_noise: float = 0.0,
    servo: bool = True,
    seed: int = 1,
) -> Pipeline:
    scene = scene or default_scene()
    model = mujoco.MjModel.from_xml_string(build_xml(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    arm = Arm(model, data)
    arm.q = HOME_Q
    data.ctrl[arm.act_adr] = HOME_Q
    mujoco.mj_forward(model, data)

    cam = SimCameraSource(model, data, "wrist_cam", 960, 720, noise_std=camera_noise)
    X_true = true_hand_eye(arm, cam)

    if use_true_hand_eye:
        X_use, he_t, he_r = X_true, 0.0, 0.0
    else:
        res, _ = calibrate_hand_eye(model, data, arm, cam, scene, seed=seed)
        X_use = res.X
        dt, dr = X_true.distance_to(X_use)
        he_t, he_r = dt * 1000, np.degrees(dr)

    # Derive the object from the TARGET LABEL rather than trusting a separate
    # index. Two sources of truth for "which object" is how the first eval run
    # reported a failure for a pick that had actually succeeded -- it measured
    # the wrong cube.
    if object_index is None:
        matches = [o for o in scene.objects if f"cube_{o.marker_id}" == target]
        if not matches:
            raise ValueError(
                f"target '{target}' matches no object in the scene; "
                f"have {[f'cube_{o.marker_id}' for o in scene.objects]}")
        obj = matches[0]
    else:
        obj = scene.objects[object_index]
    obj_start = arm.body_pose(obj.name).t.copy()

    # Vantage points covering the table, computed once at setup using whichever
    # hand-eye estimate this run is using.
    scan_qs = make_scan_qs(arm, [(0.33, 0.02), (0.30, 0.16), (0.36, -0.14)], X_use)
    if not scan_qs:
        scan_qs = (OBSERVE_Q,)

    ex = Executor(SimClock())
    ex.add(RobotNode("robot", ex.bus, ex.tf, ex.clock, model, data, arm,
                     T_tool_camera_optical=X_use, control_hz=100.0))
    ex.add(CameraNode("camera", ex.bus, ex.tf, ex.clock, cam, hz=15.0))
    ex.add(DetectorNode("detector", ex.bus, ex.tf, ex.clock,
                        ArucoDetector({o.marker_id: MarkerSpec(o.marker_size, "cube")
                                       for o in scene.objects})))
    ex.add(PoseEstimatorNode("pose", ex.bus, ex.tf, ex.clock, max_rmse_px=2.0))
    task = ex.add(PickPlaceNode("task", ex.bus, ex.tf, ex.clock, arm, TaskConfig(
        target_label=target,
        object_width=2 * obj.half,
        object_height=2 * obj.half,
        place_xy=scene.bin_pos,
        home_q=HOME_Q,
        scan_qs=scan_qs,
        use_servo=servo,
    )))
    return Pipeline(model, data, arm, cam, ex, task, obj, scene, he_t, he_r, obj_start)


def run_demo(
    scene: SceneSpec | None = None,
    target: str = "cube_1",
    object_index: int | None = None,
    use_true_hand_eye: bool = False,
    camera_noise: float = 0.0,
    servo: bool = True,
    max_sim_time: float = 45.0,
    seed: int = 1,
    verbose: bool = False,
) -> DemoResult:
    p = build_pipeline(scene, target, object_index, use_true_hand_eye,
                       camera_noise, servo, seed)
    ex, task, arm, obj, obj_start = p.executor, p.task, p.arm, p.obj, p.object_start
    he_t, he_r = p.handeye_error_mm, p.handeye_error_deg
    scene = p.scene

    ex.start()
    while not task.finished and ex.clock.now() < max_sim_time:
        ex.spin_once()

    # Attribution: what did the task BELIEVE, vs what was true?
    perception_err = float("nan")
    grasp_off = float("nan")
    est = task.log_data.committed_estimate
    if est is not None:
        # The marker sits on the top face; compare like with like by using the
        # same offset the task applied.
        true_marker = obj_start + np.array([0, 0, obj.half + 0.0024])
        perception_err = float(np.linalg.norm(est.t - true_marker)) * 1000
    if task.log_data.grasp_pose is not None:
        grasp_off = float(np.linalg.norm(
            task.log_data.grasp_pose.t[:2] - obj_start[:2])) * 1000

    obj_end = arm.body_pose(obj.name).t.copy()
    px, py = scene.bin_pos
    place_err = float(np.linalg.norm(obj_end[:2] - np.array([px, py]))) * 1000
    success = task.state is State.DONE and place_err < 75.0 and obj_end[2] > 0.005

    return DemoResult(
        success=success,
        final_state=task.state.name,
        failure_reason=task.log_data.failure_reason,
        handeye_error_mm=he_t, handeye_error_deg=he_r,
        servo_corrections_mm=[c * 1000 for c in task.log_data.servo_corrections_m],
        object_start=obj_start, object_end=obj_end,
        place_error_mm=place_err, sim_time=ex.clock.now(),
        log=ex.logs() if verbose else "",
        perception_error_mm=perception_err, grasp_offset_mm=grasp_off,
    )
