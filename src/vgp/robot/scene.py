"""MJCF builder for the workcell: 6-DoF arm, parallel gripper, wrist camera.

WHY BUILD THE ARM RATHER THAN IMPORT ONE
----------------------------------------
Importing a Franka or UR5 model is one line and hides everything. Defining the
chain ourselves means:

  - we know every link length, so forward kinematics can be checked against an
    independent calculation rather than against the thing that produced it;
  - we control the wrist geometry, so the Jacobian conditioning and the
    singularities are ours to reason about, not surprises;
  - the camera mount can be made deliberately AWKWARD (offset and tilted), so
    hand-eye calibration cannot accidentally succeed by returning identity.

That last point matters more than it sounds. A hand-eye test with the camera
mounted straight down the tool axis passes with a solver that ignores half its
input. Ours does not.

GEOMETRY
--------
UR-style 6R with a spherical wrist (last three axes intersect at one point).
A spherical wrist decouples position from orientation and keeps the Jacobian
well-conditioned away from the obvious singularities. Tabletop scale, ~0.6 m
reach.

    J1  Z   base yaw
    J2  Y   shoulder lift
    J3  Y   elbow
    J4  Y   wrist pitch      }
    J5  Z   wrist yaw        }  axes intersect -> spherical wrist
    J6  Y   wrist roll       }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[3] / "assets"

# Link lengths (metres). Named so the kinematics are readable, and used by the
# FK cross-check in the tests.
BASE_H = 0.10        # pedestal top
SHOULDER_Z = 0.08    # J1 -> J2 offset along Z
UPPER_ARM = 0.30     # J2 -> J3
FOREARM = 0.26       # J3 -> J4
WRIST_Z = 0.09       # J4 -> wrist centre (J5/J6)
FLANGE = 0.07        # wrist centre -> tool0
FINGER_LEN = 0.045   # tool0 -> fingertip (defines the grasp TCP)

JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")

# Home configuration: TCP at ~(0.42, 0, 0.14) with the gripper pointing down.
#
# NOTE j5 = pi/2, NOT zero. The wrist is Y-Z-Y (j4 about Y, j5 about Z, j6 about
# Y), which is gimbal-locked when j5 = 0: j4 and j6 axes become parallel and the
# wrist loses a degree of freedom. Measured manipulability at j5 = 0 is ~1e-16
# (a true singularity); at j5 = pi/2 it is ~5e-3.
#
# Starting IK from a singular seed is not merely slow -- the damped solver
# correctly refuses to move in the lost direction, so it stalls with a few
# degrees of orientation error that more iterations never remove.
HOME_Q = np.array([0.0, 0.60, 1.20, 1.34, np.pi / 2, 0.0])

# Observation pose: TCP at ~(0.28, 0.02, 0.38) looking down, which puts the
# WRIST camera at ~(0.34, 0.02, 0.47) -- high enough that the whole work area
# is in frame.
#
# This is a real constraint of eye-in-hand, not an implementation detail: the
# camera can only look where the arm puts it, so "search" is a MOTION, not a
# passive wait. From HOME the camera sees nothing at all (measured) because it
# sits 0.23 m up and tilted, with the objects outside its 58-degree cone. An
# eye-to-hand camera would not have this problem -- and would instead be
# occluded by the arm during the grasp. That is the actual trade between the
# two mounts.
#
# This value is COLLISION-CHECKED. The first one we computed satisfied the
# pose exactly and folded the wrist through the forearm; the arm stalled
# 2.6 rad short with no error anywhere. See Arm.in_collision.
OBSERVE_Q = np.array([-3.069316, -0.188428, -0.840304, -2.091684, 0.072315, -0.021133])

# Deliberately awkward eye-in-hand mount: offset from the tool axis AND tilted,
# so T_tool_camera is far from identity in both translation and rotation.
CAM_OFFSET = (0.055, 0.0, -0.02)     # in tool0 frame
CAM_TILT_DEG = 22.0                  # pitched to look ahead of the fingers


@dataclass
class ObjectSpec:
    """A graspable cube carrying a marker on its top face."""

    name: str
    marker_id: int
    pos: tuple[float, float, float]
    yaw: float = 0.0
    half: float = 0.025              # half-edge, so a 50 mm cube
    rgba: str = "0.85 0.25 0.2 1"

    @property
    def marker_size(self) -> float:
        # The marker PNG includes a 25% quiet zone on each side, so the black
        # marker itself occupies 1/1.5 of the textured face. Getting this wrong
        # scales every pose from this object -- with a perfect residual.
        return 2 * self.half / 1.5


@dataclass
class SceneSpec:
    objects: list[ObjectSpec] = field(default_factory=list)
    table_z: float = 0.0
    table_half: tuple[float, float] = (0.45, 0.45)
    # Static calibration marker on the table, for hand-eye. Bigger than the
    # object markers because depth uncertainty scales as 1/size (Step 3) and
    # hand-eye error propagates into EVERY later pose.
    calib_marker_id: int = 0
    calib_marker_size: float = 0.09
    calib_marker_pos: tuple[float, float] = (0.36, 0.0)
    bin_pos: tuple[float, float] = (0.10, -0.34)
    camera_fovy: float = 58.0


def _marker_assets(ids: list[int]) -> str:
    return "\n".join(
        f'    <texture name="tex{i}" type="2d" file="tag{i}.png"/>\n'
        f'    <material name="mat{i}" texture="tex{i}" texuniform="false" '
        f'texrepeat="1 1" specular="0" shininess="0"/>'
        for i in sorted(set(ids))
    )


def build_xml(spec: SceneSpec) -> str:
    ids = [o.marker_id for o in spec.objects] + [spec.calib_marker_id]
    bx, by = spec.bin_pos
    cx, cy = spec.calib_marker_pos
    cs = spec.calib_marker_size

    objects_xml = []
    for o in spec.objects:
        # Marker plate sits a hair proud of the cube's top face so it is never
        # z-fighting with it -- z-fighting renders as speckle, which destroys
        # marker decoding in a way that looks like a detector bug.
        plate = o.marker_size * 1.5 / 2
        objects_xml.append(f"""
    <body name="{o.name}" pos="{o.pos[0]} {o.pos[1]} {o.pos[2]}" euler="0 0 {o.yaw}">
      <freejoint name="{o.name}_free"/>
      <geom name="{o.name}_body" type="box" size="{o.half} {o.half} {o.half}"
            rgba="{o.rgba}" mass="0.05" friction="1.2 0.02 0.001"
            solref="0.004 1" solimp="0.95 0.99 0.001"/>
      <geom name="{o.name}_tag" type="box" pos="0 0 {o.half + 0.0012}"
            size="{plate} {plate} 0.0012" material="mat{o.marker_id}"
            contype="0" conaffinity="0" mass="0.001"/>
    </body>""")

    return f"""
<mujoco model="vgp_workcell">
  <compiler angle="radian" texturedir="{ASSETS / 'markers'}"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"
          impratio="10" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
  </visual>

  <default>
    <joint damping="2.0" armature="0.05"/>
    <geom rgba="0.55 0.57 0.60 1" friction="1 0.005 0.0001"/>
    <default class="arm">
      <position kp="900" forcerange="-200 200"/>
    </default>
    <default class="finger">
      <position kp="500" forcerange="-60 60"/>
      <geom rgba="0.2 0.2 0.22 1" friction="1.8 0.05 0.001"
            solref="0.004 1" solimp="0.97 0.995 0.001"/>
    </default>
  </default>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.35 0.40 0.48" rgb2="0.08 0.09 0.12" width="256" height="256"/>
    <texture name="tablewood" type="2d" builtin="checker" rgb1="0.62 0.60 0.56"
             rgb2="0.56 0.54 0.50" width="256" height="256"/>
    <material name="tablemat" texture="tablewood" texrepeat="6 6" specular="0.1"/>
{_marker_assets(ids)}
  </asset>

  <worldbody>
    <light name="key" pos="0.3 0.3 1.6" dir="-0.2 -0.2 -1" diffuse="0.75 0.75 0.75"
           castshadow="true"/>
    <light name="fill" pos="-0.4 -0.4 1.2" dir="0.3 0.3 -1" diffuse="0.35 0.35 0.35"
           castshadow="false"/>

    <geom name="table" type="box" pos="0.25 0 {spec.table_z - 0.02}"
          size="{spec.table_half[0]} {spec.table_half[1]} 0.02" material="tablemat"/>

    <!-- Static hand-eye calibration target -->
    <body name="calib_tag" pos="{cx} {cy} {spec.table_z + 0.0015}">
      <geom type="box" size="{cs * 1.5 / 2} {cs * 1.5 / 2} 0.0015"
            material="mat{spec.calib_marker_id}" contype="0" conaffinity="0"/>
    </body>

    <!-- Place bin -->
    <body name="bin" pos="{bx} {by} {spec.table_z}">
      <geom type="box" pos="0 0 0.001" size="0.075 0.075 0.001" rgba="0.2 0.4 0.75 1"
            contype="0" conaffinity="0"/>
      <geom type="box" pos="0.077 0 0.02" size="0.003 0.078 0.02" rgba="0.2 0.4 0.75 1"/>
      <geom type="box" pos="-0.077 0 0.02" size="0.003 0.078 0.02" rgba="0.2 0.4 0.75 1"/>
      <geom type="box" pos="0 0.077 0.02" size="0.078 0.003 0.02" rgba="0.2 0.4 0.75 1"/>
      <geom type="box" pos="0 -0.077 0.02" size="0.078 0.003 0.02" rgba="0.2 0.4 0.75 1"/>
    </body>
{''.join(objects_xml)}

    <!-- ================= ARM ================= -->
    <body name="base_link" pos="0 0 0">
      <geom name="g_base" type="cylinder" size="0.07 {BASE_H / 2}" pos="0 0 {BASE_H / 2}"/>
      <body name="link_1" pos="0 0 {BASE_H}">
        <joint name="j1" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
        <geom name="g_link1" type="cylinder" size="0.055 0.045" pos="0 0 0.03"/>
        <body name="link_2" pos="0 0 {SHOULDER_Z}">
          <joint name="j2" type="hinge" axis="0 1 0" range="-2.4 2.4"/>
          <geom name="g_link2" type="capsule" size="0.042" fromto="0 0 0  0 0 {UPPER_ARM}"/>
          <body name="link_3" pos="0 0 {UPPER_ARM}">
            <joint name="j3" type="hinge" axis="0 1 0" range="-2.7 2.7"/>
            <geom name="g_link3" type="capsule" size="0.035" fromto="0 0 0  0 0 {FOREARM}"/>
            <body name="link_4" pos="0 0 {FOREARM}">
              <joint name="j4" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
              <geom name="g_link4" type="capsule" size="0.030" fromto="0 0 0  0 0 {WRIST_Z}"/>
              <body name="link_5" pos="0 0 {WRIST_Z}">
                <joint name="j5" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
                <geom name="g_link5" type="sphere" size="0.032"/>
                <body name="link_6" pos="0 0 0">
                  <joint name="j6" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
                  <geom name="g_link6" type="cylinder" size="0.028 {FLANGE / 2}" pos="0 0 {FLANGE / 2}"/>

                  <!-- tool0: the flange frame, where the URDF chain ends -->
                  <body name="tool0" pos="0 0 {FLANGE}">
                    <site name="tool0_site" size="0.004" rgba="1 1 0 0.6"/>

                    <!-- Eye-in-hand camera. Offset AND tilted on purpose:
                         T_tool_camera must be far from identity so a
                         hand-eye solver cannot pass by ignoring its input. -->
                    <body name="camera_link" pos="{CAM_OFFSET[0]} {CAM_OFFSET[1]} {CAM_OFFSET[2]}"
                          euler="0 {np.deg2rad(CAM_TILT_DEG):.6f} 0">
                      <geom type="box" size="0.012 0.020 0.010" rgba="0.1 0.1 0.12 1"
                            contype="0" conaffinity="0" mass="0.02"/>
                      <!-- MuJoCo cameras look down their own -Z (OpenGL), so
                           zaxis="0 0 -1" makes the camera look along the BODY's
                           +Z -- i.e. roughly where the fingers point. Setting
                           zaxis="0 0 1" (the intuitive-looking choice) aims it
                           backwards, out of the robot's rear. The symptom is a
                           perfectly good image of nothing and zero detections.
                           SimCameraSource then converts to the OpenCV optical
                           convention on top of this. -->
                      <camera name="wrist_cam" pos="0 0 0" zaxis="0 0 -1"
                              fovy="{spec.camera_fovy}"/>
                    </body>

                    <!-- Parallel-jaw gripper. Two slide joints, mirrored. -->
                    <geom name="g_palm" type="box" pos="0 0 0.012" size="0.032 0.018 0.012"/>
                    <!-- Finger pads offset by their own half-width so that at
                         qpos = 0 their INNER FACES meet at x = 0 instead of
                         interpenetrating. Without the offset the closed gripper
                         is permanently self-colliding, which silently poisons
                         every collision query the planner makes. Gripper width
                         is then exactly fl + fr. -->
                    <body name="finger_left" pos="0 0 0.024">
                      <joint name="fl" type="slide" axis="1 0 0" range="0 0.035"
                             damping="8"/>
                      <geom name="g_fingerL" class="finger" type="box"
                            pos="0.006 0 {FINGER_LEN / 2}"
                            size="0.006 0.014 {FINGER_LEN / 2}"/>
                    </body>
                    <body name="finger_right" pos="0 0 0.024">
                      <joint name="fr" type="slide" axis="-1 0 0" range="0 0.035"
                             damping="8"/>
                      <geom name="g_fingerR" class="finger" type="box"
                            pos="-0.006 0 {FINGER_LEN / 2}"
                            size="0.006 0.014 {FINGER_LEN / 2}"/>
                    </body>
                    <!-- grasp: the TCP, BETWEEN the fingertips. This, not
                         tool0, is what a grasp pose refers to. -->
                    <site name="grasp_site" pos="0 0 {0.024 + FINGER_LEN}"
                          size="0.005" rgba="0 1 0 0.8"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- Links that overlap BY CONSTRUCTION. MuJoCo excludes direct parent-child
       pairs automatically, but our wrist packs link_4, link_5 and link_6 at
       nearly the same point (a spherical wrist), so several non-adjacent pairs
       are permanently in contact too.

       These must be excluded or every collision query returns True and the
       planner can never accept any configuration -- which is exactly what
       happened: the HOME pose itself reported as self-colliding. -->
  <contact>
    <exclude body1="base_link" body2="link_1"/>
    <exclude body1="link_1" body2="link_2"/>
    <exclude body1="link_3" body2="link_4"/>
    <exclude body1="link_3" body2="link_5"/>
    <exclude body1="link_4" body2="link_5"/>
    <exclude body1="link_4" body2="link_6"/>
    <exclude body1="link_4" body2="tool0"/>
    <exclude body1="link_5" body2="link_6"/>
    <exclude body1="link_5" body2="tool0"/>
    <exclude body1="link_6" body2="tool0"/>
    <exclude body1="link_6" body2="finger_left"/>
    <exclude body1="link_6" body2="finger_right"/>
    <exclude body1="tool0" body2="finger_left"/>
    <exclude body1="tool0" body2="finger_right"/>
    <exclude body1="finger_left" body2="finger_right"/>
  </contact>

  <actuator>
    <position class="arm" name="a1" joint="j1" ctrlrange="-6.28 6.28"/>
    <position class="arm" name="a2" joint="j2" ctrlrange="-2.4 2.4"/>
    <position class="arm" name="a3" joint="j3" ctrlrange="-2.7 2.7"/>
    <position class="arm" name="a4" joint="j4" ctrlrange="-6.28 6.28"/>
    <position class="arm" name="a5" joint="j5" ctrlrange="-6.28 6.28"/>
    <position class="arm" name="a6" joint="j6" ctrlrange="-6.28 6.28"/>
    <position class="finger" name="afl" joint="fl" ctrlrange="0 0.035"/>
    <position class="finger" name="afr" joint="fr" ctrlrange="0 0.035"/>
  </actuator>
</mujoco>
"""


def default_scene() -> SceneSpec:
    return SceneSpec(objects=[
        ObjectSpec("cube_a", 1, (0.33, 0.12, 0.026), yaw=0.35, rgba="0.85 0.25 0.20 1"),
        ObjectSpec("cube_b", 2, (0.40, -0.10, 0.026), yaw=-0.5, rgba="0.25 0.70 0.35 1"),
    ])
