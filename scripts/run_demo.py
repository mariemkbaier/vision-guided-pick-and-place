#!/usr/bin/env python3
"""Run one end-to-end vision-guided pick-and-place in simulation."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from vgp.task.demo import run_demo

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="cube_1")
ap.add_argument("--true-handeye", action="store_true", help="skip calibration, use ground truth")
ap.add_argument("--noise", type=float, default=0.0, help="camera noise std (grey levels)")
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()

r = run_demo(target=a.target, use_true_hand_eye=a.true_handeye,
             camera_noise=a.noise, verbose=a.verbose)
print(f"result          : {'SUCCESS' if r.success else 'FAILURE'}  ({r.final_state})")
if r.failure_reason: print(f"  reason        : {r.failure_reason}")
print(f"hand-eye error  : {r.handeye_error_mm:.3f} mm / {r.handeye_error_deg:.3f} deg")
print(f"servo corrections: {[f'{c:.1f}mm' for c in r.servo_corrections_mm]}")
print(f"object moved    : {np.round(r.object_start,3)} -> {np.round(r.object_end,3)}")
print(f"place error     : {r.place_error_mm:.1f} mm")
print(f"sim time        : {r.sim_time:.1f} s")
if a.verbose: print("\n--- log ---\n" + r.log)
