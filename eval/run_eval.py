#!/usr/bin/env python3
"""Evaluation harness: success rates and ERROR ATTRIBUTION.

WHY ATTRIBUTION IS THE WHOLE POINT
----------------------------------
"87% grasp success" is almost useless on its own. The number you can act on is
*why the other 13% failed*, and in a five-stage pipeline every failure looks
identical at the gripper: the arm goes to the wrong place.

So each condition below is designed to isolate one stage by REMOVING its error
and re-measuring. The difference between two conditions attributes error to the
thing that changed, which is the only way to know where to spend effort.

  true-handeye vs calibrated   -> how much error does Step 4 contribute?
  servo on vs off              -> what does closed-loop re-observation buy?
  noise sweep                  -> how does perception degrade, and does the
                                  task layer's gating catch it?
  randomised placements        -> is success a property of the pipeline or of
                                  one lucky object position?

Reproducibility comes free from the Step 1 decision to run single-threaded on a
SimClock: same seed, same trajectory, every time.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from vgp.robot.scene import ObjectSpec, SceneSpec
from vgp.task.demo import run_demo


def random_scene(rng: np.random.Generator) -> tuple[SceneSpec, str]:
    """One cube at a random reachable spot, plus a distractor.

    A distractor is not decoration: it is what makes "take detections[0]" fail,
    and what makes a wide-open gripper knock over a neighbour. A single-object
    benchmark hides both.
    """
    # Keep objects clear of the static calibration marker at (0.36, 0.0). A
    # cube parked on top of it makes hand-eye collect zero views and the whole
    # trial fails for a reason that has nothing to do with the pipeline -- a
    # benchmark artefact, not a result.
    CALIB_XY, CALIB_CLEAR = np.array([0.36, 0.0]), 0.11

    def sample():
        while True:
            px, py = rng.uniform(0.27, 0.43), rng.uniform(-0.17, 0.18)
            if np.linalg.norm(np.array([px, py]) - CALIB_XY) > CALIB_CLEAR:
                return px, py

    x, y = sample()
    yaw = rng.uniform(-np.pi, np.pi)
    # Distractor at least 12 cm away so the task stays well-posed.
    while True:
        dx, dy = sample()
        if np.hypot(dx - x, dy - y) > 0.12:
            break
    scene = SceneSpec(objects=[
        ObjectSpec("cube_a", 1, (x, y, 0.026), yaw=yaw, rgba="0.85 0.25 0.20 1"),
        ObjectSpec("cube_b", 2, (dx, dy, 0.026), yaw=rng.uniform(-np.pi, np.pi),
                   rgba="0.25 0.70 0.35 1"),
    ])
    return scene, "cube_1"


def run_condition(name: str, trials: int, seed0: int = 0, **kw) -> dict:
    rows = []
    for i in range(trials):
        rng = np.random.default_rng(seed0 + i)
        scene, target = random_scene(rng)
        try:
            r = run_demo(scene=scene, target=target, seed=seed0 + i, **kw)
            rows.append(r)
        except Exception as e:                       # a crash is a failure, recorded
            print(f"  trial {i}: EXCEPTION {type(e).__name__}: {str(e)[:90]}")
    if not rows:
        return {"name": name, "trials": 0}
    ok = [r for r in rows if r.success]
    fails: dict[str, int] = {}
    for r in rows:
        if not r.success:
            key = r.failure_reason.split(";")[0][:60] or f"state={r.final_state}"
            fails[key] = fails.get(key, 0) + 1
    return {
        "name": name,
        "trials": len(rows),
        "success_rate": len(ok) / len(rows),
        "place_err_mm_median": float(np.median([r.place_error_mm for r in ok])) if ok else None,
        "place_err_mm_p90": float(np.percentile([r.place_error_mm for r in ok], 90)) if ok else None,
        "perception_err_mm_median": float(np.nanmedian([r.perception_error_mm for r in rows])),
        "handeye_err_mm": float(np.nanmedian([r.handeye_error_mm for r in rows])),
        "servo_correction_mm_median": float(np.median(
            [c for r in rows for c in r.servo_corrections_mm] or [np.nan])),
        "failures": fails,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", default="eval/out/results.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    conditions = [
        ("A. ground-truth hand-eye (perception+control only)",
         dict(use_true_hand_eye=True, camera_noise=0.0)),
        ("B. calibrated hand-eye (FULL pipeline)",
         dict(use_true_hand_eye=False, camera_noise=0.0)),
        ("C. calibrated + camera noise sigma=3",
         dict(use_true_hand_eye=False, camera_noise=3.0)),
    ]
    if not a.quick:
        conditions += [
            ("D. calibrated + camera noise sigma=8",
             dict(use_true_hand_eye=False, camera_noise=8.0)),
            ("E. ABLATION: no close-range servo (open loop from first view)",
             dict(use_true_hand_eye=False, camera_noise=3.0, servo=False)),
        ]

    results = []
    for name, kw in conditions:
        print(f"\n--- {name} ({a.trials} trials) ---")
        r = run_condition(name, a.trials, **kw)
        results.append(r)
        if r["trials"]:
            print(f"  success {r['success_rate']:.0%}   "
                  f"place err median {r['place_err_mm_median']}   "
                  f"perception err {r['perception_err_mm_median']:.2f} mm   "
                  f"hand-eye {r['handeye_err_mm']:.2f} mm")
            for k, v in r["failures"].items():
                print(f"    {v}x  {k}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 92)
    print(f"{'condition':<48s} {'succ':>6s} {'place p50':>10s} {'place p90':>10s} "
          f"{'percep':>8s} {'hand-eye':>9s}")
    print("=" * 92)
    for r in results:
        if not r["trials"]:
            continue
        p50 = f"{r['place_err_mm_median']:.1f}mm" if r["place_err_mm_median"] is not None else "-"
        p90 = f"{r['place_err_mm_p90']:.1f}mm" if r["place_err_mm_p90"] is not None else "-"
        print(f"{r['name']:<48s} {r['success_rate']:>5.0%} {p50:>10s} {p90:>10s} "
              f"{r['perception_err_mm_median']:>7.2f}m {r['handeye_err_mm']:>8.2f}m")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
