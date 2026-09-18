#!/usr/bin/env python3
"""Render the pick-and-place to an MP4 -- no GUI, no permissions needed.

    ./.venv/bin/python scripts/record_demo.py

Produces a side-by-side video: a scene view on the left, and on the right the
WRIST CAMERA with live detection overlay -- the actual frames perception
consumes, not a separate render. The current task state is burned in, so you
can watch SEARCH -> SERVO -> PREGRASP -> DESCEND -> CLOSE and see what the
robot was looking at during each.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import mujoco
import numpy as np

from vgp.msgs import Header, Image
from vgp.perception.aruco import ArucoDetector, MarkerSpec
from vgp.perception.pnp import PnPFailure, solve_pnp
from vgp.task.demo import build_pipeline

GREEN, AMBER, WHITE, DARK = (80, 230, 80), (0, 190, 255), (245, 245, 245), (28, 28, 30)


def _label(img, lines, org=(14, 30), colour=WHITE, scale=0.7):
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (org[0], org[1] + int(30 * scale / 0.7) * i),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cube_1")
    ap.add_argument("--true-handeye", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="data/recordings/demo.mp4")
    ap.add_argument("--height", type=int, default=540)
    a = ap.parse_args()

    print("building pipeline (hand-eye calibration runs first)...")
    p = build_pipeline(target=a.target, use_true_hand_eye=a.true_handeye,
                       camera_noise=a.noise)
    print(f"hand-eye error: {p.handeye_error_mm:.2f} mm / {p.handeye_error_deg:.3f} deg")

    det = ArucoDetector({o.marker_id: MarkerSpec(o.marker_size, "cube")
                         for o in p.scene.objects})
    ex, task = p.executor, p.task
    ex.start()

    H = a.height
    W = int(H * 4 / 3)
    # A free camera looking at the workspace, independent of the wrist camera.
    scene_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(p.model, scene_cam)
    scene_cam.lookat[:] = [0.26, -0.06, 0.09]
    scene_cam.distance = 1.25
    scene_cam.azimuth = 138
    scene_cam.elevation = -22

    renderer = mujoco.Renderer(p.model, height=H, width=W)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             a.fps, (W * 2, H))
    if not writer.isOpened():
        print(f"ERROR: could not open a video writer for {out_path}")
        sys.exit(1)

    dt = 1.0 / a.fps
    next_frame = 0.0
    last_state = None
    frames = 0

    while not task.finished and ex.clock.now() < 60.0:
        ex.spin_once()
        if task.state != last_state:
            print(f"  {ex.clock.now():6.2f}s  {task.state.name}")
            last_state = task.state
        if ex.clock.now() < next_frame:
            continue
        next_frame += dt

        # LEFT: the scene.
        renderer.update_scene(p.data, camera=scene_cam)
        left = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(left, (0, 0), (W, 96), DARK, -1)
        cube = p.arm.body_pose(p.obj.name).t
        _label(left, [f"STATE: {task.state.name}",
                      f"t={ex.clock.now():5.2f}s   grip={p.arm.gripper_width*1000:4.1f}mm"
                      f"   cube z={cube[2]*100:4.1f}cm"], colour=AMBER, scale=0.66)

        # RIGHT: exactly what perception sees, with its own overlay.
        img, info = p.camera.read()
        right = cv2.cvtColor(img.data, cv2.COLOR_RGB2BGR)
        right = cv2.resize(right, (W, H))
        sx, sy = W / img.data.shape[1], H / img.data.shape[0]
        for d in det.detect(Image(Header(0, "camera_optical"), img.data)).detections:
            c = (d.corners * [sx, sy]).astype(int)
            cv2.polylines(right, [c], True, GREEN, 2, cv2.LINE_AA)
            cv2.putText(right, d.label, (c[0][0], max(c[0][1] - 10, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
            if d.model_points is None:
                continue
            try:
                r = solve_pnp(d.model_points, d.corners, info)
            except PnPFailure:
                continue
            txt = f"{r.pose.t[2]*100:.0f}cm  rmse {r.rmse_px:.2f}px"
            if r.ambiguous:
                txt += "  AMBIGUOUS"
            cv2.putText(right, txt, (c[0][0], min(c[2][1] + 22, H - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        AMBER if r.ambiguous else GREEN, 2, cv2.LINE_AA)
        cv2.rectangle(right, (0, 0), (W, 40), DARK, -1)
        _label(right, ["WRIST CAMERA  (what perception sees)"], colour=WHITE, scale=0.6)

        writer.write(np.hstack([left, right]))
        frames += 1

    writer.release()
    renderer.close()

    cube = p.arm.body_pose(p.obj.name).t
    px, py = p.scene.bin_pos
    err = np.linalg.norm(cube[:2] - np.array([px, py])) * 1000
    print(f"\nfinal state : {task.state.name}")
    if task.log_data.failure_reason:
        print(f"reason      : {task.log_data.failure_reason}")
    print(f"place error : {err:.1f} mm")
    print(f"\nwrote {out_path}  ({frames} frames, {frames/a.fps:.1f}s)")
    print(f"open it with:  open {out_path}")


if __name__ == "__main__":
    main()
