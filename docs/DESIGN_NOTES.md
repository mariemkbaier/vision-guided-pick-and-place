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

## 7. Step 3 — detection and 6D pose

### 7.1 Why fiducial markers first

Markers are not the impressive answer; a learned 6D pose estimator is. They are
the right answer *first* for a reason that outweighs impressiveness.

A pick-and-place pipeline has five stages that can each be wrong, and an error
at the gripper is **indistinguishable between them**. Markers give a perception
stage whose accuracy is known, bounded, and independently verifiable — so when
the arm misses, perception can be ruled *out* and the search moves to hand-eye.
Start with a learned estimator and every failure is a four-way argument.

They also give the rest of the pipeline something to be graded against: the
colour/plane path is scored against marker poses, and Step 4's hand-eye
calibration is driven by them.

### 7.2 Reprojection error is the confidence signal, not the detector score

A detector score answers "is this a marker?". It says nothing about whether the
recovered *geometry* is sane. The failure modes that hurt a robot are geometric:
a marker seen through motion blur decodes perfectly and yields a tilted pose.
Its reprojection error rises; its detection score does not.

So `PoseEstimatorNode` gates on reprojection RMSE, in pixels — a unit we can
threshold and reason about.

### 7.3 The planar two-fold ambiguity

A planar target viewed near-frontally has **two** poses that project to almost
the same image, mirrored through the marker plane. This is genuine geometry, not
numerical noise, and it is worst exactly when the marker is small, far, or
square-on — the normal tabletop case. Picking wrong flips the approach axis and
the gripper dives in tilted. The classic symptom is a pose that flickers between
two orientations frame to frame.

`SOLVEPNP_IPPE_SQUARE` via `solvePnPGeneric` returns **both** solutions; the
default `ITERATIVE` flag converges to one local minimum and gives no hint that
another existed. That silence is the problem. We compare residuals and flag when
the runner-up explains the image nearly as well. Measured error ratio vs. tilt:

| Tilt | error ratio | flagged |
|---|---|---|
| 0° | 1.09 | **yes** |
| 5° | 2.95 | **yes** |
| 15° | 11.2 | no |
| 30° | 47.8 | no |
| 45° | 52.4 | no |

Ambiguous poses are published **flagged, not dropped** — the task layer can move
the camera and look again, which perception cannot. An honest "I am not sure"
beats a confident coin flip when the consequence is a collision.

### 7.4 Monocular pose error is anisotropic — measured, not asserted

Depth is recovered from a marker's apparent *size*, so a pixel of corner noise
gives `dZ ≈ Z²/(s·fx)` against a lateral `dx ≈ Z/fx`. The ratio is **Z/s**.

Monte-Carlo over 25 noise realisations, comparing measured spread to prediction:

| Marker | Distance | measured dZ/dx | predicted Z/s |
|---|---|---|---|
| 30 mm | 0.4 m | 14.8× | 13.3× |
| 30 mm | 1.0 m | 29.6× | 33.3× |
| 50 mm | 0.7 m | 13.6× | 14.0× |

The 1/s law is near-exact (measured 1.68× vs. predicted 1.67× for 30 mm vs.
50 mm), and Z² holds. Three design consequences we actually act on:

- **Approach along the camera axis** where possible, so the badly-known
  direction is absorbed by the approach rather than becoming a lateral miss.
- **Bigger markers beat closer cameras** — `dZ ∝ 1/s` is cheaper to fix.
- **Never reach open-loop from one monocular view.** Step 6 servos.

**Sim flattery, quantified.** Absolute magnitudes came in ~20× *below*
prediction because synthetic corner refinement achieves ≈0.015 px where a real
camera gives 0.1–0.5 px. The scaling laws transfer; the absolute numbers do not.
Step 7 must report both.

### 7.5 Two conventions, two bugs

**A marker facing the camera is rotated 180° about X, not identity.** The ArUco
marker frame's +Z points *out toward the viewer*; the camera's +Z points *into
the scene*. My test fixture used identity, which renders the marker mirrored —
and a mirrored marker does not decode at all, so every single view came back
"NOT DETECTED". Caught by projecting the model points and noticing the corner
labelled TL had the *largest* image y.

**Plane offset sign depended on where the plane came from.** Lifting the support
plane by half an object's height must move it *toward* the camera. But
`offset_by(+h/2)` moves along the normal, and a normal's direction is an
accident of construction — `Plane.from_pose` takes the frame's +Z, while a
fitted plane's normal could point either way. So the same call removed the bias
in one case and **doubled** it in the other. Replaced with
`offset_toward_camera()`, which encodes the *intent* rather than the sign.

### 7.6 The bias that averaging cannot remove

For a box of height *h* on a table, the centroid of its **silhouette** is up the
object's side, not on the table. Intersecting that ray with the table surface
gives a point pulled toward the camera — proportional to *h* and to how
obliquely you look. It is **systematic, not noise**, so averaging over frames
does not remove it, and it is the right size to make a gripper clip the near
edge. Handled by lifting the plane by *h*/2, which requires knowing *h* — stated
as an explicit argument rather than hidden.

### 7.7 Plane lifting conditioning

Ray–plane intersection gives depth from a single view, but `t = c/(n·d)`
explodes as the ray flattens. Measured, 1 px of noise at image centre:

| Incidence | 3D displacement |
|---|---|
| 90° (looking down) | 0.67 mm |
| 30° | 1.33 mm |
| 8° | 4.79 mm |

Conditioning is set entirely by camera *mounting*, so no detector improvement
fixes it. `intersect()` refuses below 10° rather than returning a confident
number.

### 7.8 What the colour path deliberately cannot do

A bounding box has four numbers; a pose has six. Orientation is not recoverable
from an axis-aligned box at all — a square seen rotated has the same bbox. So
`ColorDetector` emits `model_points=None` and `pose_from_detection` **refuses**
it with an error that explains why. The constraint has to come from somewhere
explicit (the table plane), and markers are easier precisely because they carry
their own.

HSV over RGB because hue separates from brightness, so a threshold survives
illumination change. Its hard limit, stated rather than tuned around: hue is
meaningless at low saturation, so greys/whites/blacks are undetectable and a
specular highlight on a red cube reads as "not red". The honest fix is a learned
detector — the natural upgrade path.

---

## 9. Step 4 — hand-eye calibration

### 9.1 The step that makes perception actionable

Steps 2–3 give an object's pose in the **camera** frame. The robot moves in its
**base** frame. Nothing connects them, and no amount of better perception will —
a perfect 6D pose in the wrong frame is useless.

Eye-in-hand formulation: at each pose, `A_i = T_base_tool` (from FK, essentially
exact) and `B_i = T_cam_target` (from PnP). The marker does not move, so
`A_i X B_i` is constant, giving the classical `AX = XB`.

### 9.2 Why two independent solvers

We implement Park & Martin's method *and* call `cv2.calibrateHandEye`. Two
implementations with no shared code path check each other; one implementation
agreeing with itself checks nothing.

That paid for itself immediately. Our first result was 103 mm off with a 180°
rotation error — and the cross-check disagreed by *exactly* the amount we were
wrong, which localised the bug to our code rather than the data. Cause: in
`A_j⁻¹A_i · X = X · B_j B_i⁻¹`, the two sides compose **the other way round**
from each other — `A` is `(older)⁻¹(newer)` while `B` is `(older)(newer)⁻¹`.
Pattern-matching the `A` form onto `B` gives `B⁻¹`. Derive it; don't copy it.

### 9.3 Rotate about two non-parallel axes, or `X` is not identifiable

Pure geometry, not numerics. With rotations about one axis only, *any* `X`
composed with an extra rotation about that axis fits equally well — an entire
one-parameter family of perfect solutions. The solver does not fail; it returns
a family member with an excellent residual.

Measured with near-single-axis motions: **94 mm of error** in the recovered `X`,
while the residual read 7.5 mm. The Step 2 lesson in a new costume — *the
residual cannot tell you the problem was well-posed.* Hence a hard
`rotation_axis_spread` gate, computed from the data before any solve, so it can
be reported while still collecting.

### 9.4 Result

Full loop in sim — render → detect → PnP → solve, starting from a nominal mount
guess **11.2 mm / 5.6° wrong**:

| | value |
|---|---|
| Recovered `X` vs ground truth | **0.207 mm / 0.020°** |
| Target-pose scatter across 15 poses | 0.339 mm / 0.106° |
| Our solver vs OpenCV | 0.363 mm / 0.021° |

### 9.5 The bug that a "sensible default" caused

Two of eighteen views produced 127 mm and 149 mm pose errors — with residuals of
0.03 px, confidence ratios of 76, and no ambiguity flag. Nothing downstream could
have caught them.

Cause, in two parts. First, `detections[0]` is not "the marker you want" — it is
whatever the contour scan found first, and those two views saw only the *cube*
markers, not the calibration one. Second, the detector applied a **default marker
size** to unregistered ids, so a 33 mm marker was measured as 40 mm.

A guessed size does not fail. It produces a pose with a perfect reprojection
error, scaled wrong along the view ray, completely confident. The fix is Step 2's
rule again: **`default_size_m=None` refuses to guess**, and a `select(dets,
label)` helper exists so "take the first one" is never the path of least
resistance.

---

## 10. Step 5 — kinematics, planning, and control

### 10.1 Writing our own IK turned out to be the gain

MuJoCo has no IK, which forced the derivation PyBullet's one-liner would have
hidden. Damped least squares replaces each singular value σ with `σ/(σ²+λ²)`:
for σ ≫ λ that is the exact inverse, and as σ → 0 it goes to **0 instead of ∞**.
The solver smoothly gives up on directions the arm cannot move in, rather than
commanding unbounded velocity. A slightly wrong pose beats a robot flinging
itself at its velocity limit.

Three things had to be added before it actually worked, each found by measurement:

**Nullspace leak (0.42 mm → 0.0002 mm).** The projector is built from the
*damped* inverse, so it is only approximately orthogonal to the task. Far from
the target that is irrelevant; near it the leak is the same size as the residual
and the solve stalls. Fading the secondary objective out as error shrinks keeps
posture shaping where it helps and removes it where it hurts.

**Joint-limit masking.** Clamping `q` after the step is not enough — a joint *on*
its limit still has a non-zero Jacobian column, so the solver keeps allocating
motion to a joint that cannot move, and the clamp silently truncates it. Symptom:
IK stalls with one joint pinned and a few degrees of orientation error that no
number of iterations removes. Zeroing those columns tells the solver the truth.

**Collision checking.** See 10.3.

Result: **8/8 targets solved, 7–99 µm**, converging in 8–106 iterations.

### 10.2 Spend the task's symmetries before blaming the solver

A parallel-jaw gripper is 180°-symmetric about its approach axis — swapping which
finger is on which side grasps the same object identically. A square cube adds
90°, giving four equivalent yaws.

That is not cosmetic. Measured: a top-down grasp at (0.30, 0, 0.15) with yaw 0
requires a wrist roll of *exactly π* — the Y-Z-Y wrist's gimbal lock. The
equivalent grasp at yaw π needs roll 0 and is trivially conditioned. **Same
physical grasp.** Enumerating the symmetry took IK success from 5/8 to 7/8 for
free.

General lesson: when IK fails, first ask whether the *task* really required that
exact pose, or whether you over-specified it.

### 10.3 The bug that presented four layers away

An "observation pose" with j4 = 259° satisfied the target pose exactly, respected
every joint limit, and was perfectly well-conditioned — and folded the wrist
through the forearm. IK reported success, the controller commanded it, and the
arm **stopped 2.6 rad short**.

Nothing errored. The actuators saturated against the contact constraint —
measured `actuator_force` +200 N·m exactly cancelled by `qfrc_constraint`
−199.4 N·m — so the robot sat still while the controller insisted it was
commanding correctly. The failure surfaced as *"the camera cannot see the
object"*, because the arm was not where anyone thought it was.

Widening the wrist limits to ±2π (realistic for a UR-style arm) is what made
these wound-up solutions reachable, so the fix is not narrower limits — it is
checking collision at IK solutions.

Getting that check right needed two model fixes. Adjacent links overlap **by
construction** (our spherical wrist packs link 4/5/6 at nearly one point), and
the two fingers interpenetrated at qpos 0. Without the `<contact><exclude>` set
and the finger offset, *every* configuration reported as colliding — including
HOME.

### 10.4 Trapezoidal vs quintic — the actual trade

A path says *where*; a trajectory says *when*. Handing a controller bare
waypoints pushes the velocity decision somewhere invisible.

| | duration | peak v used | peak a used |
|---|---|---|---|
| Trapezoidal | 1.167 s | 76% | 98% |
| Quintic | 1.387 s | 90% | 75% |

Trapezoidal is time-optimal for the given bounds but has discontinuous
acceleration — infinite jerk at the corners. Quintic has zero jerk at the
endpoints and is **19% slower** for the same limits, because it ramps instead of
cruising. Free transit uses trapezoidal; approach and retreat near the object use
quintic.

### 10.5 Gravity compensation: 6.8 mm → 0.0 mm

A bare position servo must generate holding torque from *position error alone* —
the arm sags until `kp·error` balances gravity. Measured 6.8 mm of TCP droop at a
typical grasp configuration.

6.8 mm sounds small until you notice the gripper has ~10 mm of clearance per side
around a 50 mm cube. The fingers then descend off-centre, catch the cube's edge
and shove it aside — which presents as a *perception or calibration* error rather
than a control one. Feeding forward `qfrc_bias` (gravity + Coriolis) removes it
entirely, and is what every industrial controller does.

---

## 11. Step 6 — the task layer, and four things it refuses to do

### 11.1 Why servoing is the central design decision

Step 3 measured monocular depth uncertainty at ~`Z/s` times lateral, growing as
`Z²`. Reaching open-loop from one distant observation bets the grasp on the
worst-measured axis. So the arm moves to a stand-off, **re-observes from ~0.15 m
instead of ~0.40 m**, and only then descends. Closing the distance 2.5× improves
depth certainty ~6× — more than any detector improvement would.

### 11.2 The refusals

The task acts only on poses that are fresh, unambiguous, and low-residual. Each
refusal is logged with a reason, because "we saw it but the pose was 0.4 s old"
and "we never saw it" need completely different fixes.

The ambiguity refusal immediately proved its worth, by making the pipeline fail:
looking straight down at a planar marker is *exactly* the degenerate case, so
every frame was flagged and the arm correctly would not move.

The right response was not to relax the gate. It was to **change the viewpoint** —
28° off-axis turns an error ratio of 1.09 into 47. When an estimator is
ill-conditioned, the cheapest fix is usually the viewpoint, not the estimator.
Having the camera on a robot is precisely the freedom to do that.

### 11.3 Three bugs that only an integrated system reveals

**Search is a motion, not a wait.** With eye-in-hand the camera only sees where
the arm puts it. From HOME it sees *nothing* — the objects are outside its 58°
cone. And the search window was originally measured from when the move was
*commanded*, so it expired before the arm arrived: the task reported "never saw
the object" having looked for zero seconds. One fixed vantage point also left 25%
of randomised placements unseen; three are used now.

**The marker is not the object.** The fiducial is on the cube's *top face*, so
grasping at the measured pose put the fingertips on the top millimetre. This
*passed* the holding check — the fingers were genuinely blocked — lifted the cube
successfully, and dropped it partway through the transport move. Found by logging
the object's position per state and watching it reappear on the table 100 mm from
the bin. The fix is a known property of the object model, not something
perception can discover.

**Pose the camera, not the tool.** The camera sits 55 mm off the tool axis and
pitched 22°, so putting the *gripper* above the object puts the *camera* looking
110 mm to one side of it. At a 150 mm stand-off the object is simply out of
frame — "lost sight of the target while servoing", with nothing wrong except
aiming the wrong frame.

**And a two-stage approach is not optional.** Servoing from a tilted viewpoint
means a single Cartesian move to the grasp travels *diagonally*, straight through
the side of the object. Measured: the fingers shoved the cube 14 mm and forced
themselves open to 74 mm. Hence PREGRASP (directly above, joint-space) then
DESCEND (purely vertical, Cartesian).

### 11.4 A frame bug that the frame system was built to catch

Hand-eye necessarily produces `tool0 → camera_**optical**` (PnP's frame), but the
published TF chain is `tool0 → camera_link → camera_optical`. Inserting the
hand-eye result as the *first* hop applies the optical rotation **twice**.

It does not error. It produced object positions that were plausible — right
magnitude, wrong place: (0.73, −0.14, 0.66) for a cube actually at
(0.33, 0.12, 0.03) — and the only symptom was "no IK solution", four layers from
the cause.

The uncomfortable part: `SE3` would have caught this, and did not, because the
call site passed hand-written `parent=`/`child=` arguments that *overrode* the
labels. A safety mechanism you can bypass silently is one you will bypass.

---

## 12. Step 7 — evaluation

### 12.1 Attribution, not a single number

"87% grasp success" is nearly useless: in a five-stage pipeline every failure
looks identical at the gripper. Each condition isolates one stage by removing its
error and re-measuring — ground-truth vs calibrated hand-eye attributes Step 4's
contribution, servo on/off quantifies closed-loop re-observation, and a noise
sweep tests whether the task layer's gating actually catches degradation.

Randomised object placement plus a **distractor** matters: the distractor is what
makes `detections[0]` fail and what a too-wide gripper knocks over. A
single-object benchmark hides both.

### 12.2 A benchmark artefact, caught and fixed

The first randomised runs threw `HandEyeError: need >= 3 poses` — a cube had
landed on the calibration marker. That is a property of the *benchmark*, not of
the pipeline, and counting it as a pipeline failure would have been dishonest.
Scene generation now keeps objects clear of the target.

Likewise the harness initially measured the wrong cube (a hardcoded index
alongside a target label — two sources of truth) and reported a failure for a
pick that had succeeded.

---

## 13. Open questions / known limitations

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
  is a poor stand-in for the real failure modes. Quantified in 7.4: synthetic
  corner noise is ~0.015 px against 0.1–0.5 px for a real camera, so sim
  flatters absolute accuracy by roughly 20×. Step 7 must report both.
- **No temporal filtering of poses.** Each frame is solved independently. A
  static object would benefit from averaging, and a moving one from a constant
  velocity filter — but note (7.6) that the plane-lift bias is systematic and
  would survive any amount of averaging.
- **Ambiguity is flagged but not yet resolved.** The task layer is supposed to
  request a second viewpoint. Until Step 6 exists, an ambiguous pose simply
  carries a flag nobody acts on.
- **`AMBIGUITY_RATIO = 3.0` is judgement, not measurement.** Chosen
  conservatively (a false alarm costs one camera move; a missed ambiguity costs
  a collision). The measured ratios in 7.3 separate cleanly at ±15°, so the
  exact value matters little in that regime — but it has not been swept the way
  the Step 2 gates were.
- **The colour detector is a demo, not a product.** See 7.8: it cannot handle
  uncoloured objects at all, and its HSV bands need per-cell tuning.
- **No collision-aware path PLANNING** — only collision *checking* at IK
  solutions and along Cartesian waypoints. There is no RRT: a straight line
  through an obstacle is detected and refused, never routed around. That is the
  largest single gap between this and a production cell.
- **Grasp pose selection is top-down only.** Real bin picking needs grasp
  synthesis over arbitrary approach directions; `grasp.py` raises on anything
  but `-z`.
- **No force feedback during the grasp.** Success is inferred from finger
  position (they stopped short, so something is between them), which is what a
  real gripper's position feedback gives — but a force/torque sensor would catch
  a slipping object mid-transport, which we currently cannot.
- **Dynamics are unmodelled in planning.** Trajectories respect joint velocity
  and acceleration limits but not payload-dependent torque limits; carrying a
  heavier object would need re-timing.
- **Hand-eye is calibrated once, offline.** Real cells need to detect when it has
  drifted (a knock to the camera mount). The target-scatter residual would show
  it, but nothing monitors it online.
- **The `SE3` frame check can be bypassed** by passing explicit `parent`/`child`
  at a call site, which is exactly how the double-optical-rotation bug (11.4)
  slipped through. A stricter API would make overriding labels harder.
