"""End-to-end task tests: the pipeline as a whole, and its refusals.

These are slow (they step physics and render) but they are the only tests that
check the thing we actually built. Everything else checks a component.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vgp.robot.scene import ObjectSpec, SceneSpec
from vgp.task.demo import run_demo
from vgp.task.pick_place import State


def _scene(x=0.33, y=0.12, yaw=0.35):
    return SceneSpec(objects=[
        ObjectSpec("cube_a", 1, (x, y, 0.026), yaw=yaw, rgba="0.85 0.25 0.20 1"),
        ObjectSpec("cube_b", 2, (0.40, -0.10, 0.026), yaw=-0.5, rgba="0.25 0.70 0.35 1"),
    ])


@pytest.fixture(scope="module")
def run_true():
    return run_demo(scene=_scene(), target="cube_1", use_true_hand_eye=True)


@pytest.fixture(scope="module")
def run_calibrated():
    return run_demo(scene=_scene(), target="cube_1", use_true_hand_eye=False)


@pytest.mark.slow
class TestEndToEnd:
    def test_succeeds_with_ground_truth_hand_eye(self, run_true):
        assert run_true.success, f"{run_true.final_state}: {run_true.failure_reason}"

    def test_succeeds_with_calibrated_hand_eye(self, run_calibrated):
        """The FULL pipeline: calibrate, perceive, plan, grasp, place -- with no
        privileged information anywhere in the loop."""
        r = run_calibrated
        assert r.success, f"{r.final_state}: {r.failure_reason}"

    def test_object_actually_moves_to_the_bin(self, run_true):
        assert run_true.place_error_mm < 40

    def test_hand_eye_calibration_is_accurate(self, run_calibrated):
        assert run_calibrated.handeye_error_mm < 5.0
        assert run_calibrated.handeye_error_deg < 1.0

    def test_perception_error_is_millimetric(self, run_calibrated):
        assert run_calibrated.perception_error_mm < 8.0

    def test_grasp_is_centred_on_the_object(self, run_true):
        """REGRESSION: the marker is on the object's TOP FACE, so grasping at
        the measured pose put the fingertips on the top millimetre of the cube.
        It lifted, then dropped the cube mid-transport."""
        assert run_true.grasp_offset_mm < 12.0

    def test_servo_makes_a_correction(self, run_true):
        """If this is ever zero, close-range re-observation is not happening
        and the pipeline is silently open-loop."""
        assert run_true.servo_corrections_mm
        assert run_true.servo_corrections_mm[0] > 0.0

    def test_reaches_done(self, run_true):
        assert run_true.final_state == State.DONE.name

    def test_second_object_also_works(self):
        r = run_demo(scene=_scene(), target="cube_2", use_true_hand_eye=True)
        assert r.success, f"{r.final_state}: {r.failure_reason}"


@pytest.mark.slow
class TestDeterminism:
    def test_identical_runs_give_identical_results(self):
        """The property the single-threaded executor was chosen to buy. Without
        it the eval numbers mean nothing."""
        a = run_demo(scene=_scene(), target="cube_1", use_true_hand_eye=True)
        b = run_demo(scene=_scene(), target="cube_1", use_true_hand_eye=True)
        assert np.allclose(a.object_end, b.object_end)
        assert a.place_error_mm == b.place_error_mm
        assert a.sim_time == b.sim_time


@pytest.mark.slow
class TestRefusals:
    """The pipeline must fail LOUDLY rather than act on bad information."""

    def test_unreachable_object_fails_cleanly(self):
        """An object far outside the workspace must be reported, not
        approximated."""
        r = run_demo(scene=SceneSpec(objects=[
            ObjectSpec("cube_a", 1, (0.90, 0.60, 0.026)),
            ObjectSpec("cube_b", 2, (0.40, -0.10, 0.026)),
        ]), target="cube_1", use_true_hand_eye=True, max_sim_time=25.0)
        assert not r.success
        assert r.failure_reason, "a failure must carry a reason"

    def test_target_not_in_scene_raises(self):
        with pytest.raises(ValueError, match="matches no object"):
            run_demo(scene=_scene(), target="cube_9", use_true_hand_eye=True)

    def test_survives_camera_noise(self):
        r = run_demo(scene=_scene(), target="cube_1", use_true_hand_eye=True,
                     camera_noise=6.0)
        # Either it succeeds, or it refuses with a reason. Silently placing the
        # cube somewhere wrong is the outcome we will not accept.
        assert r.success or r.failure_reason
