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

## 3. Open questions / known limitations

- *(to be filled as we build — each step appends here)*
