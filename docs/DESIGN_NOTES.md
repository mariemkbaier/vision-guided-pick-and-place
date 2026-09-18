# Design Notes — Vision-Guided Pick-and-Place

A running log of *decisions and their tradeoffs*. Read this before the code.
Every entry answers: what did we pick, what was the alternative, what did
choosing cost us.

---

## 0. Problem statement

Given a camera looking at a table with objects on it, command a 6-DoF arm to
pick a named object and place it in a target bin — with no human in the loop,
and with the perception path identical whether frames come from a simulator or
a real camera.

The pipeline, and the coordinate frame each stage lives in:

```
 frame:        image px        camera 3D        robot base        joint space
              ┌─────────┐    ┌───────────┐    ┌───────────┐    ┌────────────┐
 camera ─────▶│ detect  │───▶│  6D pose  │───▶│ transform │───▶│ IK + plan  │──▶ grasp
              └─────────┘    └───────────┘    └───────────┘    └────────────┘
                 2D bbox        PnP on          T_base_cam        damped least
                 + corners      known model     (hand-eye)        squares
```

The hard part is not any single box. It is that **an error in any one of them
is indistinguishable from an error in the others** at the gripper. Step 7's
eval harness exists to attribute error to the right stage.

---

## 1. Dependency decisions

### 1.1 Simulator: MuJoCo, not PyBullet  *(decided 2026-09-18)*

**Chose:** `mujoco>=3.2`.

**Why:** PyBullet was the first choice — it ships a built-in IK solver and is
the classic teaching simulator. It does not install here. There is no arm64
wheel for macOS 26 / Python 3.12, so pip builds from source, and PyBullet's
*vendored copy of zlib* is old enough that its `#define`s collide with the
macOS 26.2 SDK:

```
MacOSX26.2.sdk/usr/include/_stdio.h:322:7: error: expected identifier or '('
```

That is a bug in PyBullet's third-party tree, not something a flag fixes.

**Cost of the switch:** MuJoCo has **no built-in IK**. PyBullet's
`calculateInverseKinematics()` would have been one call. We now implement
damped-least-squares IK ourselves (Step 5).

**Why that cost is acceptable — arguably a gain:** a hidden IK call is a black
box in an interview. A Jacobian-based solver we wrote is something we can
derive on a whiteboard, and it lets us do nullspace posture control and joint
limit handling that PyBullet's solver hides. MuJoCo is also the current
industry/research default (DeepMind), has first-class offscreen rendering, and
gives us an analytic Jacobian for free via `mj_jac`.

### 1.2 Vision: OpenCV 4.14, pinned `<5`  *(decided 2026-09-18)*

**Chose:** `opencv-contrib-python>=4.10,<5`.

**Why the cap:** OpenCV 5.0.0 installs fine and `cv2.aruco` works, but the
`cv2.calibrateHandEye()` Python binding is **missing** — while the
`cv2.CALIB_HAND_EYE_TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS` enums are still
exported. A binding gap in a fresh major release. Step 4 needs that function as
a cross-check oracle, so we stay on the stable line.

**Why contrib and not plain:** `cv2.aruco` moved to contrib in 4.7.

**Cost:** we are one major version behind. Acceptable — 4.x is what production
vision stacks run, and the 5.x bindings will mature.

### 1.3 Python 3.12, not 3.14

The system default here is 3.14. Scientific wheels (and MuJoCo) are reliably
built for 3.12 on arm64; 3.14 would have pushed more packages to source builds.
Nothing in this project needs 3.13+ language features.

---

## 2. Architecture decision: ROS-shaped, without ROS

**Chose:** our own `Node` / `Bus` / `TFTree` runtime in `src/vgp/runtime/`.

**Why:** ROS 2 is the right answer in industry and the wrong answer on this
machine — no ROS install, no Docker, and ARM-Mac ROS 2 + Gazebo is a
multi-hour yak shave that teaches nothing about vision. But the *mental model*
ROS enforces — independent nodes, typed messages on topics, a transform tree
as the single source of truth for geometry — is exactly what keeps a
perception-to-control pipeline from becoming a tangle of implicit assumptions.

So we build that model, small and explicit, and keep the node bodies free of
runtime specifics.

**Cost:** no rviz, no rosbag, no MoveIt, no community nodes. We reimplement
small versions of the first two (`viz/`, recordings).

**Exit path:** porting to `rclpy` should mean rewriting `runtime/` and leaving
`nodes` and `msgs` almost untouched. If a port would require touching node
bodies, we have leaked runtime detail into them — that is the design invariant
to defend.

---

## 3. Step 1 — the runtime spine

### 3.1 One naming rule for transforms

`T_a_b` maps points from frame `b` into frame `a`; `p_a = T_a_b @ p_b`. Read
"a from b". Composition then cancels like fractions — `T_a_b @ T_b_c = T_a_c` —
and the same object is *also* the pose of `b` expressed in `a`. Both readings
are true, and being able to say why is worth more than memorising the rule.

**Enforced, not just documented:** `SE3` carries optional `parent`/`child`
labels and raises `FrameMismatch` when inner frames disagree. Composing
`T_base_cam @ T_tag_cam` is a plain matrix multiply in numpy — it returns
garbage silently. Now it raises.

**Cost:** we gave up raw 4×4 arrays, which compose with `@` and pass straight
to cv2/MuJoCo. `.as_matrix()` is the one-call escape hatch, deliberately
explicit so library boundaries stay greppable.

### 3.2 Quaternion order is in the function name

scipy/ROS use `(x,y,z,w)`; MuJoCo uses `(w,x,y,z)`. There is no `from_quat` —
only `from_quat_xyzw` and `from_quat_wxyz`. Verbose on purpose: you cannot pass
the wrong order without typing the wrong word.

### 3.3 `camera_link` vs `camera_optical`

Same physical point, different axis conventions: robotics is X-forward/Y-left/
Z-up, OpenCV is Z-forward/X-right/Y-down. A fixed non-identity rotation sits
between them permanently. Forgetting it gives a pose at a plausible distance
but rotated ~90°, in a direction that changes as the camera moves — people
"fix" it with a fudge factor and never recover. It is a named constant with
per-axis tests.

### 3.4 Tree, not graph

One path between any two frames, so the answer is unique. A general graph
permits two paths that disagree — which is exactly what happens once you have
both a hand-eye calibration and a fiducial estimate of the same camera pose.
`set_transform` enforces single-parent so you must *choose* which is
authoritative. `detach()` makes deliberate re-parenting (object → gripper on
grasp) explicit.

### 3.5 Time buffering, and refusing to extrapolate

Encoders run ~500 Hz, PnP ~15 Hz, and a PnP result is stamped when its *image*
was captured — 30–80 ms in the past. Resolving it needs where the arm was *then*.
On a moving arm that differs by centimetres, so the TF buffer interpolates
between bracketing samples. This is the difference between a pipeline that
works only when stationary and one that works while moving.

Lookups past the newest sample **raise** rather than falling back to "latest".
Extrapolating robot state is guessing about the future position of a heavy
machine; failing lets the caller wait or abort. tf2 made the same call.

### 3.6 Single-threaded executor

**Gained:** bit-reproducible runs. Same seed, same inputs, same trajectory —
which is what makes the Step 7 grasp-success number mean anything, and what
turns a failure into something reproducible rather than a story.

**Lost:** no real parallelism; throughput is bounded by the slowest callback.
Mitigated by counting timer overruns, so "the detector missed 12% of its 15 Hz
deadlines" stays observable even though nothing ran concurrently.

**Exit path:** a thread-pool executor changes `runtime/executor.py` only —
*provided* nodes obeyed the no-shared-mutable-state rule.

### 3.7 Bug found and fixed: timer deadline drift

The executor advanced deadlines with `next_due += period`. Since 0.01 is not
exactly representable in binary floating point, 100 accumulations give
`1.0000000000000007` — so a 100 Hz timer spun for exactly 1.0 s fired **99**
times.

What makes it worth recording: a 10 Hz timer accumulates to
`0.9999999999999999`, erring the *other* way, so it fired the correct 10. The
bug **passed the 10 Hz test and failed only at 100 Hz**. A single-rate test
suite would have shipped it, and it would have resurfaced much later as an
off-by-one in a frame count where nobody suspects the clock.

**Fix:** keep an integer tick index, derive `start + n * period`. Error is
bounded at one rounding instead of growing with n — what real-time schedulers
do. Plus a 1 ns epsilon at the spin boundary so one ulp does not decide whether
the final tick fires. Regression test sweeps 14 rates from 1–500 Hz.

---

## 4. Open questions / known limitations

- **No inter-node parallelism.** See 3.6. Fine for one laptop; would need
  revisiting for a real multi-rate control stack.
- **Bus fan-out is synchronous**, so a slow subscriber blocks its publisher.
  Acceptable given single-threading, but it means callback cost is a latency
  budget item, not just a throughput one.
- **TF interpolation is linear** (LERP + SLERP). For a fast-moving arm between
  widely spaced samples this under-models the actual path; at 500 Hz encoder
  rates the error is negligible.
- *(later steps append here)*
