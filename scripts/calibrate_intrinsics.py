#!/usr/bin/env python3
"""Interactive intrinsic calibration from the webcam.

The interesting part is not calling cv2.calibrateCamera -- it is COACHING the
operator to collect data that satisfies the gates in calib/intrinsics.py.

The three failure modes are all data-collection failures, invisible in the
final residual, and all fixable only while the camera is still running:
  - not enough tilt  -> focal length nearly unidentifiable
  - no edge corners  -> distortion coefficients unconstrained
  - poor coverage    -> both of the above, mildly

So the live view reports what is still MISSING, and auto-capture only accepts a
view that adds something new. Telling someone "your calibration was bad" after
they put the board away is useless; telling them "tilt it more, now" is not.

    python scripts/calibrate_intrinsics.py --square-size-m 0.025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgp.calib.intrinsics import (
    CalibrationResult, CheckerboardSpec, IntrinsicCalibrator, grid_occupancy,
    recommend_board, save_intrinsics,
)
from vgp.sensors.webcam import WebcamSource


def _tilt_deg(corners: np.ndarray, spec: CheckerboardSpec, K: np.ndarray) -> float:
    """Rough board tilt from a provisional K -- only used for live coaching."""
    ok, rvec, _ = cv2.solvePnP(spec.object_points(), corners, K, None)
    if not ok:
        return 0.0
    R, _ = cv2.Rodrigues(rvec)
    return float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--cols", type=int, default=0, help="0 = match sensor aspect")
    ap.add_argument("--rows", type=int, default=0, help="0 = match sensor aspect")
    ap.add_argument("--square-size-m", type=float, default=0.025,
                    help="MEASURED size of a printed square, in metres")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--target-views", type=int, default=20)
    ap.add_argument("--out", default="data/calib/webcam_intrinsics.yaml")
    a = ap.parse_args()

    cam = WebcamSource(a.device, a.width, a.height)
    # Pick the board AFTER opening the camera, so it matches the resolution we
    # actually got (which may not be the one we asked for -- see webcam.py).
    if a.cols and a.rows:
        spec = CheckerboardSpec(a.cols, a.rows, a.square_size_m)
    else:
        spec = recommend_board(cam.width, cam.height, a.square_size_m)
        print(f"board auto-matched to sensor aspect: {spec.cols}x{spec.rows} inner corners")
        print(f"  (generate it: python scripts/make_target.py --cols {spec.cols} "
              f"--rows {spec.rows} --square-mm {a.square_size_m * 1000:.0f})")
    cal = IntrinsicCalibrator(spec)
    print(f"capture resolution: {cam.width}x{cam.height}")
    print("SPACE=capture   a=toggle auto   c=calibrate   q=quit\n")

    W, H = cam.width, cam.height
    # Provisional K for live tilt readout only -- assumes a ~60 deg FOV. Never
    # used in the actual solve, which estimates K from scratch.
    K_guess = np.array([[W * 0.9, 0, (W - 1) / 2], [0, W * 0.9, (H - 1) / 2], [0, 0, 1]])
    auto, tilts, pts, corner_sets = True, [], [], []

    try:
        while True:
            rgb = cam.read_raw()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (spec.cols, spec.rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
            )
            view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            novel = False
            if found:
                cv2.drawChessboardCorners(view, (spec.cols, spec.rows), corners, found)
                c = corners.reshape(-1, 2)
                t = _tilt_deg(corners, spec, K_guess)
                # "Novel" = this view's centre or tilt differs enough from every
                # view already captured. Twenty near-identical views constrain
                # no more than one; diversity is the whole currency here.
                ctr = c.mean(0)
                novel = all(
                    np.linalg.norm(ctr - p) > 0.10 * W or abs(t - tt) > 8.0
                    for p, tt in zip(pts, tilts)
                ) if pts else True

            # Live coaching, using the SAME metric the final gates use --
            # otherwise the tool can say "looking good" and then reject.
            cov = edge = 0.0
            if corner_sets:
                cov, edge = grid_occupancy(np.concatenate(corner_sets), W, H)
            span = (max(tilts) - min(tilts)) if len(tilts) > 1 else 0.0
            G = CalibrationResult
            need = []
            if len(tilts) < a.target_views:
                need.append(f"{a.target_views - len(tilts)} more views")
            if span < G.MIN_TILT_SPAN_DEG:
                need.append(f"MORE TILT ({span:.0f}/{G.MIN_TILT_SPAN_DEG:.0f} deg)")
            if edge < G.MIN_EDGE_COVERAGE:
                need.append(f"BOARD TO THE EDGES ({edge:.0%}/{G.MIN_EDGE_COVERAGE:.0%} border)")
            elif cov < G.MIN_COVERAGE:
                need.append(f"SPREAD OUT ({cov:.0%}/{G.MIN_COVERAGE:.0%} cells)")

            for i, line in enumerate([
                f"views {len(tilts)}  tilt {span:.0f}d  cells {cov:.0%}  border {edge:.0%}  auto={'ON' if auto else 'off'}",
                ("NEED: " + ", ".join(need)) if need else "ALL GATES MET - press c to calibrate",
            ]):
                cv2.putText(view, line, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255) if (i and need) else (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("intrinsic calibration", view)
            key = cv2.waitKey(1) & 0xFF

            if found and ((auto and novel) or key == ord(" ")):
                if cal.add_view(rgb):
                    pts.append(corners.reshape(-1, 2).mean(0))
                    corner_sets.append(corners.reshape(-1, 2))
                    tilts.append(_tilt_deg(corners, spec, K_guess))
                    print(f"captured view {cal.n_views}  tilt={tilts[-1]:.0f}deg")
            if key == ord("a"):
                auto = not auto
            if key == ord("q"):
                return
            if key == ord("c") and cal.n_views >= 3:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()

    result = cal.calibrate()
    print("\n" + result.summary())
    save_intrinsics(a.out, result)
    print(f"\nsaved -> {a.out}")
    if not result.ok:
        print("\nSaved anyway, but the quality gates flagged problems above.")
        print("Fix the DATA and recalibrate -- a low RMSE does not rescue a")
        print("degenerate view set (see docs/DESIGN_NOTES.md section 5).")


if __name__ == "__main__":
    main()
