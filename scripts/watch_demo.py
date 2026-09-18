#!/usr/bin/env python3
"""Watch the pick-and-place in an interactive 3D window.

    ./.venv/bin/mjpython scripts/watch_demo.py

NOTE THE LAUNCHER: `mjpython`, not `python`. MuJoCo's interactive viewer has to
own the macOS main thread to talk to Cocoa, and the `mjpython` shim (shipped
with the mujoco wheel) is what arranges that. Running this with plain `python`
on macOS fails with a main-thread error.

Drag to orbit, scroll to zoom, right-drag to pan. The wrist-camera view is
shown in a second OpenCV window -- that is the image perception actually sees,
not a separate render.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from vgp.msgs import Header, Image
from vgp.perception.aruco import ArucoDetector, MarkerSpec
from vgp.task.demo import build_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cube_1")
    ap.add_argument("--true-handeye", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed; 1.0 = real time")
    ap.add_argument("--no-camera-window", action="store_true")
    a = ap.parse_args()

    print("building pipeline (hand-eye calibration runs first)...")
    p = build_pipeline(target=a.target, use_true_hand_eye=a.true_handeye,
                       camera_noise=a.noise)
    print(f"hand-eye error: {p.handeye_error_mm:.2f} mm / {p.handeye_error_deg:.3f} deg")
    print("opening viewer -- drag to orbit, scroll to zoom, ESC to quit\n")

    det = ArucoDetector({o.marker_id: MarkerSpec(o.marker_size, "cube")
                         for o in p.scene.objects})
    ex, task = p.executor, p.task
    ex.start()

    last_state = None
    wall0 = time.time()
    with mujoco.viewer.launch_passive(p.model, p.data) as viewer:
        # Frame the workspace rather than the default wide shot.
        viewer.cam.lookat[:] = [0.25, -0.05, 0.10]
        viewer.cam.distance = 1.4
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        while viewer.is_running() and not task.finished and ex.clock.now() < 60.0:
            # Advance simulated time no faster than `speed` x wall time, so the
            # motion is watchable. The pipeline itself is unaffected -- it runs
            # on the SimClock and does not know or care about wall time.
            target_sim = (time.time() - wall0) * a.speed
            while ex.clock.now() < target_sim and not task.finished:
                ex.spin_once()

            if task.state != last_state:
                cube = p.arm.body_pose(p.obj.name).t
                print(f"  {ex.clock.now():6.2f}s  {task.state.name:<10s} "
                      f"TCP={np.round(p.arm.fk_grasp().t, 3)}  "
                      f"grip={p.arm.gripper_width*1000:5.1f}mm  "
                      f"cube={np.round(cube, 3)}")
                last_state = task.state

            viewer.sync()

            if not a.no_camera_window:
                img, info = p.camera.read()
                view = cv2.cvtColor(img.data, cv2.COLOR_RGB2BGR)
                for d in det.detect(Image(Header(0, "camera_optical"), img.data)).detections:
                    c = d.corners.astype(int)
                    cv2.polylines(view, [c], True, (0, 255, 0), 2)
                    cv2.putText(view, d.label, tuple(c[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(view, f"WRIST CAMERA  |  {task.state.name}", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.imshow("wrist camera (what perception sees)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    cube = p.arm.body_pose(p.obj.name).t
    px, py = p.scene.bin_pos
    err = np.linalg.norm(cube[:2] - np.array([px, py])) * 1000
    print(f"\nfinal state : {task.state.name}")
    if task.log_data.failure_reason:
        print(f"reason      : {task.log_data.failure_reason}")
    print(f"cube moved  : {np.round(p.object_start, 3)} -> {np.round(cube, 3)}")
    print(f"place error : {err:.1f} mm")


if __name__ == "__main__":
    main()
