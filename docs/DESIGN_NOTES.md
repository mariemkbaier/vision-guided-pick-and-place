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

## 5. Step 2 — camera sources and intrinsic calibration

### 5.1 One `CameraSource` interface, and ground truth kept off it

Sim and webcam implement the same `read() -> (Image, CameraInfo)`. If the sim
path went through different code, every sim result would carry an asterisk —
you could never tell whether a real-world failure was the environment or the
untested branch.

Privileged simulator state (exact camera pose, exact object poses) is exposed
on a **separate** `ground_truth()` channel, never through `read()`. If it
flowed through the normal message path a perception node could consume it — by
accident or by a well-meaning shortcut — and every accuracy number afterwards
would measure nothing. Physical separation makes that impossible rather than
merely discouraged.

`read()` returns intrinsics with *every* frame rather than once at startup.
Autofocus shifts fx/fy; any resize rescales all four of fx, fy, cx, cy. Data
and the calibration that interprets it travel together or they drift apart.

### 5.2 Two conventions verified against rendered pixels, not against argument

A sign error in a camera convention is *self-consistent*: the image renders
fine, PnP reports low reprojection error, and only the robot is wrong. So both
were checked against actual pixels.

**Principal point.** In OpenCV a pixel coordinate is its *centre*, so an image
of W pixels spans `[-0.5, W-0.5]` and the centre is `(W-1)/2`, not `W/2`.
Measured with a marker on the optical axis:

| Hypothesis | Predicted | Error vs. rendered |
|---|---|---|
| `(W-1)/2` → 319.5 | `[319.5, 239.5]` | **0.000 px** |
| `W/2` → 320.0 | `[320.0, 240.0]` | 0.707 px |

Half a pixel sounds ignorable. It is a *systematic* bias that PnP converts into
a constant pose offset, later misattributed to hand-eye calibration.

**MuJoCo/OpenGL → OpenCV.** MuJoCo looks down −Z with +Y up; OpenCV looks down
+Z with +Y down. The mapping is `diag(1,-1,-1)` — a 180° rotation about X, det
+1. (`diag(1,1,-1)` is the natural typo and is a *reflection*, which `SE3`
rejects.) Verified by checking that world +Y renders *above* image centre and
world +X to the right.

`fovy → fy = (H/2)/tan(fovy/2)` reprojects to 0.02–0.06 px against rendered
markers — that residual is centroid estimation on a finite sphere, not model
error.

### 5.3 The headline result: RMSE cannot tell you a calibration is good

Photograph a planar board face-on and focal length trades off against distance:
a board twice as far with twice the focal length makes nearly the same image.
The optimiser lands somewhere on that valley and reports a beautiful residual.

Measured against a known `fx = 780`, 20 views each:

| Views | Reprojection RMSE | fx error |
|---|---|---|
| Face-on only | 0.039 px | **28.2 %** |
| Tilted ≤ 35° | 0.043 px *(worse!)* | **0.04 %** |

**The well-conditioned calibration has the higher residual.** You cannot
distinguish these by looking at the number everyone reports. This is why every
gate below is on the **data**, not on the residual.

Threshold set by sweeping tilt span, 5 seeds each:

| Tilt span | median fx err | worst fx err |
|---|---|---|
| 0.0° | 6.7 % | 7.8 % |
| 5.2° | 0.3 % | 1.0 % |
| 10.4° | 0.2 % | **24.4 %** |
| 26.1° | 0.02 % | 0.08 % |

The median recovers by ~6°, but the **worst case stays erratic until ~26°**.
We gate on the worst case: a calibration you cannot trust to be good is not
good. Hence `MIN_TILT_SPAN_DEG = 25`.

### 5.4 Two bugs found, both silent

**Distortion applied with the removal map.** `render_view` used
`initUndistortRectifyMap`, which answers the *reverse* question (for each
ideal pixel, where to sample in a distorted source — what you want when
rectifying a real photo). Feeding an ideal render through it applies the
*inverse* distortion. It does not crash; it produces a plausibly barrel-warped
image, and the calibrator then faithfully recovered `k1 = +0.15` for a true
`−0.15`. **Sign-flipped, not noisy** — which is why the test now asserts on the
sign explicitly. Fixed by building the map from `undistortPoints` over the
output grid.

**A quality metric that punished collecting more data.** `edge_coverage` was
the *fraction of corners* in the outer margin. Going 15 → 40 views:

| Views | hull coverage | edge corner-fraction |
|---|---|---|
| 15 | 52 % | 4.0 % |
| 30 | 57 % | 2.3 % |
| 40 | 58 % | 2.6 % |

Coverage improved while "edge coverage" got *worse*, because each extra view
adds mostly central corners and dilutes the ratio. Had we gated on it, we would
have coached operators to stop early. Replaced with **grid-cell occupancy**
(fraction of 8×6 cells, and of border cells, containing ≥1 corner), which is
monotone in effort — a new view can fill a cell, never empty one. A convex hull
was rejected for a related reason: it measures *extent*, so a ring of corners
scores 100 % with an empty middle.

### 5.5 Board aspect ratio is a property of the target, not the operator

`findChessboardCorners` is all-or-nothing, so every view must fit **entirely**
in frame — quiet zone included. An aspect mismatch therefore wastes the
difference on *every single view*, and no amount of technique recovers it.
Measured on a 960×540 (16:9) sensor over 30 visibility-checked views:

| Board | bordered aspect | cell coverage | border cells |
|---|---|---|---|
| 9×6 | 1.38 | 52 % | 2.3 % |
| 11×6 | 1.62 | 58 % | 3.9 % |
| 12×7 | 1.56 | 60 % | 4.4 % |

The classic 9×6 board is 3:2 — fine on an old 4:3 camera, a permanent handicap
on a 16:9 one. `recommend_board()` picks a matched shape; the target generator
and the calibration CLI both default to it.

### 5.6 Visibility-aware pose sampling

The test fixture originally jittered board position laterally, which silently
clipped **~45 % of views** (measured) — and the survivors were the central,
low-coverage ones. Worst of both, and invisible: you ask for 20 views, get 11,
and the calibration is unstable for a reason that never appears in its output.
`pose_grid` now rejection-samples against the projected board hull, which is
also what a human does: slide the board toward the edge until it is about to
leave frame, then stop.

### 5.7 What we refuse to do

**No guessed intrinsics for a real camera.** We could infer a plausible K from
a typical FOV. A guessed K yields poses that are confidently wrong with a low
residual — far worse than an error message. Sim gets exact intrinsics for free;
real cameras must earn theirs.

**`square_size_m` is the single scale reference** for the whole pipeline. Get
it 10 % wrong and every object is 10 % too far away, permanently, with perfect
residuals — the geometry stays self-consistent, just uniformly scaled. Hence
the target generator prints the size on the board, warns when it will not fit
the paper (because "fit to page" silently rescales), and tells you to measure
the print with a ruler.

---

## 6. Open questions / known limitations

- **No inter-node parallelism.** See 3.6. Fine for one laptop; would need
  revisiting for a real multi-rate control stack.
- **Bus fan-out is synchronous**, so a slow subscriber blocks its publisher.
  Acceptable given single-threading, but it means callback cost is a latency
  budget item, not just a throughput one.
- **TF interpolation is linear** (LERP + SLERP). For a fast-moving arm between
  widely spaced samples this under-models the actual path; at 500 Hz encoder
  rates the error is negligible.
- **Checkerboard, not ChArUco.** ChArUco tolerates partial occlusion, so the
  board could leave the frame and still contribute — which would lift the
  coverage ceiling (5.5) considerably. Checkerboard was chosen for a simpler
  first pass; ChArUco is the clear upgrade and we already depend on
  `cv2.aruco` for Step 3.
- **Rolling shutter is unmodelled.** The webcam almost certainly has one. For a
  static calibration board it does not matter; for pose estimation on a moving
  object it introduces a skew our pinhole model cannot represent.
- **Sim renders are noiseless pinholes** — zero distortion, perfect focus, no
  motion blur. `SimCameraSource(noise_std=...)` adds Gaussian noise, but that
  is a poor stand-in for the real failure modes. Sim numbers will flatter us
  and Step 7 must say so.
- *(later steps append here)*
