#!/usr/bin/env python3
"""Live ArUco detection and 6D pose on your REAL camera.

    ./.venv/bin/python scripts/live_camera.py

This is the same perception code the simulated robot runs -- `ArucoDetector`
and `solve_pnp`, unchanged. That is the whole point of the `CameraSource`
abstraction: sim and real share one code path, so a result here says something
about the pipeline rather than about a separate demo branch.

WHAT TO POINT IT AT
-------------------
You do not need a printer. Open one of the generated markers on your phone and
hold the screen up to the camera:

    open assets/markers/tag1.png

Then measure the marker's black square on the phone screen with a ruler and
pass it as --marker-mm. That number is the ONLY thing setting absolute scale --
get it 10% wrong and every distance reported here is 10% wrong, with no
symptom (see docs/DESIGN_NOTES.md section 5.7).

ON INTRINSICS
-------------
6D pose needs a calibrated camera. Without one this shows DETECTION only, which
still demonstrates the detector but cannot give you metres.

`--approx-intrinsics` guesses K from a typical field of view so you can see
pose axes immediately. It is deliberately labelled APPROX in red on every
frame, and the distances it prints can easily be 10-20% wrong. The robot path
refuses guessed intrinsics outright (sensors/webcam.py note 4) -- this flag
exists for looking, not for measuring. Run scripts/calibrate_intrinsics.py for
real numbers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from vgp.msgs import CameraInfo, Header, Image
from vgp.perception.aruco import ArucoDetector, MarkerSpec
from vgp.perception.pnp import PnPFailure, solve_pnp

GREEN, AMBER, RED, WHITE = (80, 230, 80), (0, 190, 255), (60, 60, 255), (240, 240, 240)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--marker-mm", type=float, default=40.0,
                    help="MEASURED size of the marker's black square, in mm")
    ap.add_argument("--calibration", default="data/calib/webcam_intrinsics.yaml")
    ap.add_argument("--approx-intrinsics", action="store_true",
                    help="guess K so pose axes render (NOT metrically trustworthy)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.device)
    if not cap.isOpened():
        print("ERROR: could not open the camera.\n"
              "On macOS the first run needs permission:\n"
              "  System Settings > Privacy & Security > Camera > enable your\n"
              "  terminal (Terminal.app, iTerm, or Visual Studio Code).\n"
              "Quit and relaunch the terminal after granting it.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    # Read back the ACTUAL mode -- the driver may ignore the request, and a
    # calibration for the wrong resolution is a large focal-length error.
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size_m = a.marker_mm / 1000.0
    # Detector registered for ids 0-9 at the given size; unregistered ids are
    # still DETECTED but produce no pose, by design.
    det = ArucoDetector({i: MarkerSpec(size_m, "tag") for i in range(10)})

    info = None
    mode = "DETECTION ONLY (no intrinsics)"
    cal_path = Path(a.calibration)
    if cal_path.exists():
        from vgp.calib.intrinsics import load_intrinsics
        ci = load_intrinsics(cal_path)
        if (ci.width, ci.height) != (W, H):
            ci = ci.scaled(W / ci.width, H / ci.height)
        info = ci
        mode = f"CALIBRATED ({cal_path.name})"
    elif a.approx_intrinsics:
        # ~60 degree horizontal FOV is typical of a laptop webcam.
        f = W / (2 * np.tan(np.deg2rad(60) / 2))
        K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
        info = CameraInfo(Header(0.0, "camera_optical"), K, np.zeros(5), W, H)
        mode = "APPROX INTRINSICS - distances NOT trustworthy"

    print(f"camera {a.device}: {W}x{H}   marker {a.marker_mm:.0f} mm   {mode}")
    print("keys:  q quit   s save a snapshot\n")
    if info is None:
        print("No calibration found. Showing detection only.")
        print("  For pose axes now : rerun with --approx-intrinsics")
        print("  For real numbers  : ./.venv/bin/python scripts/calibrate_intrinsics.py\n")

    fps, t_prev, saved = 0.0, time.time(), 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("camera read failed (unplugged, or another app grabbed it)")
                break
            # cv2 gives BGR; our internal convention is RGB. Convert at the
            # boundary, once -- mixing them reads as bad lighting, not an error.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = det.detect(Image(Header(time.monotonic(), "camera_optical"), rgb)).detections

            view = bgr.copy()
            for d in dets:
                c = d.corners.astype(int)
                cv2.polylines(view, [c], True, GREEN, 2, cv2.LINE_AA)
                for pt in c:
                    cv2.circle(view, tuple(pt), 4, AMBER, -1)
                cv2.putText(view, d.label, (c[0][0], c[0][1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)

                if info is None or d.model_points is None:
                    continue
                try:
                    r = solve_pnp(d.model_points, d.corners, info)
                except PnPFailure:
                    continue
                cv2.drawFrameAxes(view, info.K, info.dist,
                                  cv2.Rodrigues(r.pose.R)[0], r.pose.t, size_m * 0.7, 3)
                # The two numbers that matter, from Step 3: reprojection error
                # is the confidence signal (not the detector score), and the
                # ambiguity flag warns that a near-frontal planar marker has a
                # second, mirrored solution that fits almost as well.
                lines = [
                    f"{r.pose.t[2]*100:5.1f} cm away",
                    f"rmse {r.rmse_px:4.2f} px",
                ]
                col = GREEN
                if r.ambiguous:
                    lines.append(f"AMBIGUOUS (ratio {r.error_ratio:.1f}) - tilt it")
                    col = AMBER
                for k, ln in enumerate(lines):
                    cv2.putText(view, ln, (c[0][0], c[0][1] + 22 + 24 * k),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.rectangle(view, (0, 0), (W, 72), (24, 24, 24), -1)
            cv2.putText(view, f"{len(dets)} marker(s)   {fps:4.1f} fps   {W}x{H}",
                        (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
            cv2.putText(view, mode, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        RED if "APPROX" in mode or "no intrinsics" in mode else GREEN,
                        2, cv2.LINE_AA)
            cv2.imshow("vgp - live camera (q quit, s save)", view)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                out = Path("data/recordings") / f"snap_{saved:03d}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), view)
                print(f"saved {out}")
                saved += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
