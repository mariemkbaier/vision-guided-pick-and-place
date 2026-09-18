"""Runtime tests: bus, TF tree, clock, executor."""

import numpy as np
import pytest

from vgp.msgs import Header, CameraInfo, Image
from vgp.runtime import (
    Bus, TopicTypeError, SimClock, Rate, Node, TFTree, TFLookupError, Executor,
)
from vgp.utils.transforms import SE3
from vgp.utils.frames import WORLD, BASE, CAMERA_LINK, CAMERA_OPTICAL, T_CAMERA_LINK_FROM_OPTICAL


def _info(stamp=0.0):
    return CameraInfo(Header(stamp, CAMERA_OPTICAL), np.eye(3), np.zeros(5), 8, 8)


class TestBus:
    def test_fanout_in_subscription_order(self):
        """Deterministic ordering is what makes whole runs reproducible."""
        bus = Bus(); bus.declare("/t", CameraInfo)
        seen = []
        for i in range(3):
            bus.subscribe("/t", lambda m, i=i: seen.append(i), owner=f"n{i}")
        bus.publish("/t", _info())
        assert seen == [0, 1, 2]

    def test_wrong_type_blames_the_publisher(self):
        bus = Bus(); bus.declare("/t", CameraInfo)
        with pytest.raises(TopicTypeError, match="expects CameraInfo"):
            bus.publish("/t", "nope")

    def test_undeclared_topic_raises(self):
        with pytest.raises(KeyError):
            Bus().publish("/ghost", _info())

    def test_redeclare_with_different_type_raises(self):
        bus = Bus(); bus.declare("/t", CameraInfo)
        with pytest.raises(TopicTypeError):
            bus.declare("/t", Image)

    def test_latched_replays_to_late_subscriber(self):
        """The startup race that latching exists to kill: intrinsics are
        published once, and a node that starts later must still get them."""
        bus = Bus(); bus.declare("/info", CameraInfo, latched=True)
        bus.publish("/info", _info(1.0))
        got = []
        bus.subscribe("/info", got.append, owner="late")
        assert len(got) == 1 and got[0].header.stamp == 1.0

    def test_unlatched_does_not_replay(self):
        bus = Bus(); bus.declare("/img", CameraInfo, latched=False)
        bus.publish("/img", _info())
        got = []
        bus.subscribe("/img", got.append, owner="late")
        assert got == []


class TestTFTree:
    def _tree(self):
        tf = TFTree()
        tf.set_transform(SE3.from_rt(np.eye(3), [0, 0, 0.1], parent=WORLD, child=BASE), static=True)
        tf.set_transform(SE3.from_euler("z", [0.5], [0.3, 0.0, 0.4],
                                        parent=BASE, child=CAMERA_LINK), static=True)
        tf.set_transform(T_CAMERA_LINK_FROM_OPTICAL, static=True)
        return tf

    def test_multi_hop_lookup_matches_manual_chain(self):
        tf = self._tree()
        got = tf.lookup(WORLD, CAMERA_OPTICAL)
        manual = (tf.lookup(WORLD, BASE) @ tf.lookup(BASE, CAMERA_LINK)
                  @ tf.lookup(CAMERA_LINK, CAMERA_OPTICAL))
        assert np.allclose(got.as_matrix(), manual.as_matrix(), atol=1e-12)

    def test_lookup_is_invertible_both_directions(self):
        """lookup(A,B) and lookup(B,A) must be exact inverses, from one stored
        edge. If the reverse-traversal branch were wrong, this catches it."""
        tf = self._tree()
        fwd, rev = tf.lookup(WORLD, CAMERA_OPTICAL), tf.lookup(CAMERA_OPTICAL, WORLD)
        assert np.allclose((fwd @ rev).as_matrix(), np.eye(4), atol=1e-12)

    def test_lookup_labels_result_correctly(self):
        T = self._tree().lookup(WORLD, CAMERA_OPTICAL)
        assert (T.parent, T.child) == (WORLD, CAMERA_OPTICAL)

    def test_identity_for_same_frame(self):
        assert np.allclose(self._tree().lookup(BASE, BASE).as_matrix(), np.eye(4))

    def test_disconnected_frame_raises_with_helpful_message(self):
        tf = self._tree()
        tf.set_transform(SE3.identity("floating_a", "floating_b"), static=True)
        with pytest.raises(TFLookupError, match="no transform path"):
            tf.lookup(BASE, "floating_b")

    def test_single_parent_rule_enforced(self):
        """Two parents = two disagreeing paths. Refused, not silently accepted."""
        tf = self._tree()
        with pytest.raises(ValueError, match="already has parent"):
            tf.set_transform(SE3.identity(WORLD, CAMERA_LINK), static=True)

    def test_detach_allows_deliberate_reparent(self):
        """Used for real when a grasped object moves from camera to gripper."""
        tf = self._tree()
        tf.detach(CAMERA_LINK)
        tf.set_transform(SE3.identity(WORLD, CAMERA_LINK), static=True)
        assert tf.can_lookup(WORLD, CAMERA_LINK)

    def test_rejects_unlabelled_transform(self):
        with pytest.raises(ValueError, match="labelled"):
            TFTree().set_transform(SE3.identity())

    def test_rejects_self_loop(self):
        with pytest.raises(ValueError, match="self-loop"):
            TFTree().set_transform(SE3.identity(BASE, BASE), static=True)

    # ------------------------------------------------- the time dimension

    def test_interpolates_between_samples(self):
        """The detail that lets the pipeline work on a MOVING arm: a pose
        stamped in the past must be resolved against past robot state."""
        tf = TFTree()
        for t, x in [(1.0, 0.0), (2.0, 1.0)]:
            tf.set_transform(SE3.from_rt(np.eye(3), [x, 0, 0], parent=BASE, child="obj"), stamp=t)
        assert np.isclose(tf.lookup(BASE, "obj", stamp=1.25).t[0], 0.25)

    def test_interpolates_rotation_too(self):
        tf = TFTree()
        for t, a in [(0.0, 0.0), (1.0, np.pi / 2)]:
            tf.set_transform(SE3.from_euler("z", [a], [0, 0, 0], parent=BASE, child="obj"), stamp=t)
        assert np.isclose(tf.lookup(BASE, "obj", stamp=0.5).rotation_angle(), np.pi / 4, atol=1e-9)

    def test_refuses_to_extrapolate_forward(self):
        tf = TFTree()
        tf.set_transform(SE3.identity(BASE, "obj"), stamp=1.0)
        with pytest.raises(TFLookupError, match="Refusing to extrapolate"):
            tf.lookup(BASE, "obj", stamp=5.0)

    def test_refuses_samples_predating_buffer(self):
        tf = TFTree(buffer_s=1.0)
        for t in np.arange(0.0, 5.0, 0.1):
            tf.set_transform(SE3.identity(BASE, "obj"), stamp=float(t))
        with pytest.raises(TFLookupError, match="predates"):
            tf.lookup(BASE, "obj", stamp=0.05)

    def test_buffer_is_bounded(self):
        """Unbounded history is a slow leak in a long-running process."""
        tf = TFTree(buffer_s=1.0)
        for t in np.arange(0.0, 60.0, 0.01):
            tf.set_transform(SE3.identity(BASE, "obj"), stamp=float(t))
        assert len(tf._edges[(BASE, "obj")].stamps) < 150

    def test_static_ignores_stamp(self):
        tf = self._tree()
        a = tf.lookup(WORLD, BASE, stamp=0.0)
        b = tf.lookup(WORLD, BASE, stamp=999.0)
        assert np.allclose(a.as_matrix(), b.as_matrix())


class TestClockAndRate:
    def test_sim_clock_is_deterministic(self):
        c = SimClock(); c.advance(0.5); c.advance(0.25)
        assert c.now() == 0.75

    def test_sim_clock_refuses_to_go_backwards(self):
        with pytest.raises(ValueError):
            SimClock().advance(-1.0)

    def test_rate_does_not_drift(self):
        """Absolute deadlines, not now+period -- 100 ticks land exactly on 1.0s."""
        c = SimClock(); r = Rate(c, 100.0)
        for _ in range(100):
            r.sleep()
        assert np.isclose(c.now(), 1.0, atol=1e-9)

    def test_rate_counts_overruns_instead_of_hiding_them(self):
        c = SimClock(); r = Rate(c, 100.0)
        for _ in range(10):
            c.advance(0.05)      # body takes 50ms, 5x the 10ms budget
            r.sleep()
        assert r.missed_deadlines == 10 and r.overrun_fraction == 1.0


class _Counter(Node):
    def __init__(self, *a, hz=10.0, **kw):
        super().__init__(*a, **kw)
        self.hz, self.ticks = hz, 0

    def on_configure(self):
        self.declare("/tick", CameraInfo)
        self.create_timer(self.hz, self._tick)

    def _tick(self):
        self.ticks += 1
        self.publish("/tick", _info(self.now()))


class _Listener(Node):
    def on_configure(self):
        self.heard = []
        self.subscribe("/tick", CameraInfo, self.heard.append)


class TestExecutor:
    def test_runs_timers_at_the_right_rate(self):
        ex = Executor(SimClock())
        c = ex.add(_Counter("counter", ex.bus, ex.tf, ex.clock, hz=10.0))
        ex.spin(duration=1.0)
        assert c.ticks == 10, f"expected 10 ticks in 1s at 10Hz, got {c.ticks}"

    def test_multirate_nodes_interleave_correctly(self):
        ex = Executor(SimClock())
        fast = ex.add(_Counter("fast", ex.bus, ex.tf, ex.clock, hz=100.0))
        slow = ex.add(_Counter("slow", ex.bus, ex.tf, ex.clock, hz=10.0))
        ex.spin(duration=1.0)
        assert (fast.ticks, slow.ticks) == (100, 10)

    def test_messages_reach_subscribers(self):
        ex = Executor(SimClock())
        ex.add(_Counter("counter", ex.bus, ex.tf, ex.clock, hz=10.0))
        lis = ex.add(_Listener("listener", ex.bus, ex.tf, ex.clock))
        ex.spin(duration=1.0)
        assert len(lis.heard) == 10

    def test_two_phase_startup_prevents_lost_first_message(self):
        """Every node configures before any node starts, so a publish in
        on_start() cannot outrun a subscription made in on_configure()."""
        order = []

        class A(Node):
            def on_configure(self): order.append("cfg:A")
            def on_start(self): order.append("start:A")

        class B(Node):
            def on_configure(self): order.append("cfg:B")
            def on_start(self): order.append("start:B")

        ex = Executor(SimClock())
        ex.add(A("a", ex.bus, ex.tf, ex.clock)); ex.add(B("b", ex.bus, ex.tf, ex.clock))
        ex.start()
        assert order == ["cfg:A", "cfg:B", "start:A", "start:B"]

    def test_cannot_add_nodes_after_start(self):
        ex = Executor(SimClock()); ex.start()
        with pytest.raises(RuntimeError):
            ex.add(_Counter("late", ex.bus, ex.tf, ex.clock))

    def test_run_is_bit_reproducible(self):
        """The property the whole single-threaded decision was made to buy."""
        def run():
            ex = Executor(SimClock())
            ex.add(_Counter("a", ex.bus, ex.tf, ex.clock, hz=30.0))
            ex.add(_Counter("b", ex.bus, ex.tf, ex.clock, hz=7.0))
            lis = ex.add(_Listener("l", ex.bus, ex.tf, ex.clock))
            ex.spin(duration=2.0)
            return [m.header.stamp for m in lis.heard]
        assert run() == run()

    def test_shutdown_runs_in_reverse_order(self):
        order = []

        class N(Node):
            def on_shutdown(self): order.append(self.name)

        ex = Executor(SimClock())
        for nm in ("first", "second", "third"):
            ex.add(N(nm, ex.bus, ex.tf, ex.clock))
        ex.start(); ex.shutdown()
        assert order == ["third", "second", "first"]


class TestTimerDriftRegression:
    """Regression guard for a real bug found building Step 1.

    The executor advanced deadlines with `next_due += period`. Because 0.01 is
    not exactly representable in binary floating point, 100 accumulations gave
    1.0000000000000007, so a 100 Hz timer spun for exactly 1.0 s fired 99 times.
    A 10 Hz timer accumulated to 0.9999999999999999 and fired the correct 10 --
    so the bug passed the 10 Hz test and failed only at 100 Hz.

    Sweeping many rates is the point: a single-rate test is exactly what let
    this through the first time.
    """

    @pytest.mark.parametrize("hz", [1, 2, 3, 7, 10, 15, 24, 30, 50, 60, 100, 125, 200, 500])
    def test_exact_tick_count_across_rates(self, hz):
        ex = Executor(SimClock())
        c = ex.add(_Counter("c", ex.bus, ex.tf, ex.clock, hz=float(hz)))
        ex.spin(duration=1.0)
        assert c.ticks == hz, f"{hz} Hz for 1.0 s produced {c.ticks} ticks"

    @pytest.mark.parametrize("hz", [7, 30, 100])
    def test_no_drift_over_long_run(self, hz):
        """60 s is where accumulated error would be ~60x larger."""
        ex = Executor(SimClock())
        c = ex.add(_Counter("c", ex.bus, ex.tf, ex.clock, hz=float(hz)))
        ex.spin(duration=60.0)
        assert c.ticks == hz * 60

    def test_deadlines_stay_on_the_exact_grid(self):
        ex = Executor(SimClock())
        ex.add(_Counter("c", ex.bus, ex.tf, ex.clock, hz=100.0))
        lis = ex.add(_Listener("l", ex.bus, ex.tf, ex.clock))
        ex.spin(duration=1.0)
        stamps = np.array([m.header.stamp for m in lis.heard])
        expected = np.arange(1, 101) * 0.01
        assert np.abs(stamps - expected).max() < 1e-12
