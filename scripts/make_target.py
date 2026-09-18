#!/usr/bin/env python3
"""Generate a printable checkerboard at an exact physical size.

WHY A SCRIPT AND NOT A DOWNLOADED PDF
-------------------------------------
`square_size_m` is the ONLY thing that sets absolute scale for the entire
pipeline. Every distance downstream -- object depth, grasp position, the
hand-eye baseline -- is proportional to it. A 10% error there means every
object is 10% too far away, permanently, and NOTHING in the residuals shows it:
reprojection error stays perfect because the geometry is self-consistent, just
uniformly scaled.

So the target is generated at a known DPI with the square size printed on it,
and the instructions tell you to measure the printed result with a ruler rather
than trust the printer. Printers rescale ("fit to page") and that is exactly
the silent 4% error you will never find by staring at code.

Usage:
    python scripts/make_target.py --cols 9 --rows 6 --square-mm 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DPI = 300
MM_PER_INCH = 25.4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=0, help="INNER corners across (0 = from --sensor)")
    ap.add_argument("--rows", type=int, default=0, help="INNER corners down (0 = from --sensor)")
    ap.add_argument("--sensor", default="1280x720", help="target sensor, e.g. 1280x720")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out", default="assets/checkerboard.png")
    ap.add_argument("--paper", default="A4", choices=["A4", "A3", "letter", "none"])
    a = ap.parse_args()

    if not (a.cols and a.rows):
        # A board's aspect ratio should match the sensor's. findChessboardCorners
        # is all-or-nothing, so every view must fit ENTIRELY in frame -- an
        # aspect mismatch wastes the difference on EVERY view, and no amount of
        # operator skill recovers it. Measured on 960x540: a 3:2 board (the
        # classic 9x6) reaches 52% cell coverage where a matched board reaches
        # 60%+, and edge corners -- which is what constrains distortion -- are
        # the first thing you lose.
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
        from vgp.calib.intrinsics import recommend_board

        sw, sh = (int(v) for v in a.sensor.lower().split("x"))
        rec = recommend_board(sw, sh, a.square_mm / 1000.0)
        a.cols, a.rows = rec.cols, rec.rows
        print(f"sensor {sw}x{sh} (aspect {sw/sh:.2f}) -> board {a.cols}x{a.rows} "
              f"(bordered aspect {(a.cols+2)/(a.rows+2):.2f})")

    px = int(round(a.square_mm / MM_PER_INCH * DPI))
    nx, ny = a.cols + 1, a.rows + 1        # inner corners -> squares
    margin = px                            # one-square white quiet zone
    W, H = nx * px + 2 * margin, ny * px + 2 * margin + int(0.5 * px)
    img = np.full((H, W), 255, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                y, x = margin + j * px, margin + i * px
                img[y : y + px, x : x + px] = 0

    label = f"{a.cols}x{a.rows} inner corners | square = {a.square_mm:.1f} mm | print at 100%, NO scaling"
    cv2.putText(img, label, (margin, H - int(0.15 * px)),
                cv2.FONT_HERSHEY_SIMPLEX, px / 90.0, 0, max(1, px // 40), cv2.LINE_AA)

    # Paper fit. A target that does not fit the paper gets printed "fit to page",
    # which rescales it -- and a rescaled board means square_size_m is wrong,
    # which biases EVERY distance downstream by the same factor, invisibly.
    PAPER_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0), "letter": (279.4, 215.9)}
    w_mm, h_mm = W / DPI * MM_PER_INCH, H / DPI * MM_PER_INCH
    if a.paper != "none":
        pw, ph = PAPER_MM[a.paper]                       # landscape
        if w_mm > pw - 10 or h_mm > ph - 10:
            fit_sq = a.square_mm * min((pw - 10) / w_mm, (ph - 10) / h_mm)
            print(f"WARNING: {w_mm:.0f}x{h_mm:.0f} mm does NOT fit {a.paper} "
                  f"({pw:.0f}x{ph:.0f} mm landscape).")
            print(f"  Your printer will offer 'fit to page' -- do NOT accept it. A")
            print(f"  rescaled board makes square_size_m wrong, which biases every")
            print(f"  distance in the pipeline by the same factor, with no symptom.")
            print(f"  Instead re-run with --square-mm {fit_sq:.0f} (or fewer cols/rows,")
            print(f"  or --paper A3).")
            print()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"wrote {out}  ({W}x{H}px at {DPI} dpi = "
          f"{W / DPI * MM_PER_INCH:.1f} x {H / DPI * MM_PER_INCH:.1f} mm)")
    print()
    print("BEFORE CALIBRATING:")
    print("  1. Print at 100% scale -- turn OFF 'fit to page'.")
    print(f"  2. MEASURE a printed square with a ruler. Expect {a.square_mm:.1f} mm.")
    print("     If it differs, pass the MEASURED value as --square-size-m; do not")
    print("     assume the printer obeyed. A 4% scale error is invisible in the")
    print("     reprojection residual and shifts every object pose by 4%.")
    print("  3. Tape it FLAT to something rigid. A curled sheet is not a plane,")
    print("     and planarity is the assumption the whole solve rests on.")
    print()
    print("WHILE CALIBRATING, the three things that actually matter:")
    print("  TILT the board (>=25 deg spread). Face-on views leave focal length")
    print("     nearly unidentifiable -- measured 28% focal error at a 0.04 px")
    print("     residual. This is the one people skip.")
    print("  EDGES: push the board into all four corners of the frame. Distortion")
    print("     is ~zero at the centre, so centre-only data cannot constrain it.")
    print("  CLOSE: fill the frame. Distance variety beats more views from afar.")


if __name__ == "__main__":
    main()
