#!/usr/bin/env python3
"""
Full scene + keyboard control (Genesis version): 10 tables + 9 letters + 3 cutlery items + 1 controllable robot

This is the Genesis port of scripts/scenes/scene_robot_keyboard.py (Isaac Lab).
It essentially combines two already-ported Genesis examples:
  * The scene layout (tables/letters/cutlery) comes from scene_robot_tables.py
  * The base kinematics + keyboard control come from keyboard_control.py

The biggest difference from the static display scene (scene_robot_tables.py):
  here the robot base must be free (fixed=False) so the drive wheels can push it
  around the scene. The robot starts at the top center (0, 4.5); drive it between
  the tables with WASD/QE.

Keys (WASD clashes with Genesis viewer shortcuts, so translation uses the arrow
keys and rotation uses Q/E):
  ↑/↓ forward/back, ←/→ strafe left/right, Q/E rotate in place left/right,
  combinable (e.g. ↑+←), ESC to quit.

Dependency: pip install pynput  (global keyboard listener, same as the original script)

Coordinate / quaternion convention: both Genesis and Isaac Lab use Z-up, meters,
quaternion w-x-y-z, so poses are carried over verbatim.
"""

import math
import os

import numpy as np
import genesis as gs
from pynput import keyboard

########################## base kinematics constants ##########################
WHEEL_RADIUS_M = 0.05
LINEAR_SPEED_MPS = 0.5
ANGULAR_SPEED_RADPS = 1.2
MAX_WHEEL_SPEED_RADPS = 18.0
STOP_EPS = 1.0e-4
STEERING_FULL_SPEED_ERROR_RAD = math.radians(8.0)
STEERING_ZERO_SPEED_ERROR_RAD = math.radians(35.0)
HEADING_HOLD_KP = 2.0
HEADING_HOLD_KD = 0.35
MAX_HEADING_COMP_RADPS = 0.8

# Each drive module: (steering joint, drive-wheel joint, body-frame x, body-frame y).
# ROS convention: +x forward, +y left.
DRIVE_MODULES = (
    ("tmrv0_2_joint_0", "tmrv0_2_joint_1", 0.3, -0.2),
    ("tmrv0_2_joint_2", "tmrv0_2_joint_3", -0.3, 0.2),
)

########################## init ##########################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_DIR = os.path.join(SCRIPT_DIR, "mjcf")
URDF_PATH = os.path.join(SCRIPT_DIR, "urdf", "mobile_fr3_duo_v0_2_franka_hand.urdf")

gs.init(backend=gs.gpu)
# gs.init(backend=gs.amdgpu)
# gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        # Tilted top-down view that captures the whole table grid and the robot
        camera_pos=(0.0, -7.0, 14.0),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
        max_FPS=60,
    ),
    # Small step size improves stability (same as keyboard_control.py)
    sim_options=gs.options.SimOptions(dt=0.005, gravity=(0.0, 0.0, -9.81)),
    show_viewer=True,
)

########################## ground ##########################
# Greatly increase friction to give the drive wheels enough grip
# (original script: static=2.0 / dynamic=1.5)
scene.add_entity(
    gs.morphs.Plane(),
    material=gs.materials.Rigid(friction=2.0),
)

########################## tables (11) ##########################
# The table was converted from table_edit.usd to MJCF. The conversion baked the
# 90° (about X) orientation into the body quat, so we only pass pos here. The body
# has no joint => welded to the world frame by default (equivalent to fixed).
table_mjcf = os.path.join(SCRIPT_DIR, "mjcf", "table_edit", "table_edit.xml")

# Left column of 5 (X=-2.0) + right column of 5 (X=2.0) + 1 at the bottom center.
# The top center (0, 4.5) is reserved for the robot.
table_positions = [
    (-2.0, 3.0, 0.0), (-2.0, 1.5, 0.0), (-2.0, 0.0, 0.0), (-2.0, -1.5, 0.0), (-2.0, -3.0, 0.0),
    (2.0, 3.0, 0.0), (2.0, 1.5, 0.0), (2.0, 0.0, 0.0), (2.0, -1.5, 0.0), (2.0, -3.0, 0.0),
    (0.0, -4.5, 0.0),  # bottom center
]
for pos in table_positions:
    scene.add_entity(
        gs.morphs.MJCF(file=table_mjcf, pos=pos),
    )

########################## letters (9, black) ##########################
# The letter assets were converted from *_edit.usd to MJCF (mjcf/<L>_edit/<L>_edit.xml).
# The MJCF body already bakes in each one's orientation, but the letters still need
# an extra 90° rotation about X to stand upright (matching the quat composed in the
# original USD scene); this morph.quat is composed on top of the body's existing
# orientation. The body has no joint => welded (equivalent to fixed).
# Color is forced to black via surface.
LETTER_BLACK = gs.surfaces.Default(color=(0.0, 0.0, 0.0))
LETTER_QUAT = (0.7071, 0.7071, 0.0, 0.0)  # 90° about X
letter_table_pos = {
    "A": (-2.0, 1.5, 0.7), "B": (2.0, 1.5, 0.7),
    "C": (-2.0, 0.0, 0.7), "D": (2.0, 0.0, 0.7),
    "E": (-2.0, -1.5, 0.7), "F": (2.0, -1.5, 0.7),
    "G": (-2.0, -3.0, 0.7), "H": (2.0, -3.0, 0.7),
    "I": (0.0, -4.5, 0.7),
}
for letter, (tx, ty, tz) in letter_table_pos.items():
    letter_mjcf = os.path.join(MJCF_DIR, f"{letter}_edit", f"{letter}_edit.xml")
    scene.add_entity(
        gs.morphs.MJCF(file=letter_mjcf, pos=(tx, ty, tz + 0.061), quat=LETTER_QUAT),
        surface=LETTER_BLACK,
    )

########################## cutlery (3 items) ##########################
# The cutlery assets were converted to MJCF (mjcf/<item>/<item>.xml); their orientation
# is likewise baked into the body quat, so we only pass pos.
# The original Isaac Lab offsets placed the cutlery off/below the table top (the z
# datum was 0.0 instead of the table top at 0.7), so they fell to the floor. Here we
# raise the datum to the table-top height 0.7 and pull the x/y offsets in toward the
# table center.
ikea_table_pos = (-2.0, 3.0, 0.7)  # center of the cutlery table top
cutlery_configs = {
    "bowl": {"offset": (0.12, 0.0, 0.15), "color": (1.0, 0.0, 0.0)},
    "plate": {"offset": (-0.12, 0.0, 0.13), "color": (1.0, 1.0, 0.0)},
    "spoon": {"offset": (0.0, 0.15, 0.13), "color": (0.0, 0.0, 1.0)},
}
for item, cfg in cutlery_configs.items():
    ox, oy, oz = cfg["offset"]
    item_pos = (ikea_table_pos[0] + ox, ikea_table_pos[1] + oy, ikea_table_pos[2] + oz)
    item_mjcf = os.path.join(MJCF_DIR, item, f"{item}.xml")
    scene.add_entity(
        gs.morphs.MJCF(file=item_mjcf, pos=item_pos),
        surface=gs.surfaces.Default(color=cfg["color"]),
    )

########################## robot (controllable, top center) ##########################
# The base must be free (fixed=False) so the wheels can push it around.
robot = scene.add_entity(
    gs.morphs.URDF(
        file=URDF_PATH,
        pos=(0.0, 4.5, 0.0),
        fixed=False,
    ),
    material=gs.materials.Rigid(friction=2.0),
)

########################## build ##########################
scene.build()


# ---- joint / dof handles ----
def dof(name):
    return robot.get_joint(name).dofs_idx_local[0]


# Arms (dual arm 7+7) and grippers (dual hand 2+2)
arm_joint_names = [f"{side}_fr3v2_joint{i}" for side in ("left", "right") for i in range(1, 8)]
finger_joint_names = [f"{side}_fr3v2_finger_joint{j}" for side in ("left", "right") for j in (1, 2)]
arm_dofs = [dof(n) for n in arm_joint_names]
finger_dofs = [dof(n) for n in finger_joint_names]

# Active base joints
steering_dofs = [dof(m[0]) for m in DRIVE_MODULES]
drive_dofs = [dof(m[1]) for m in DRIVE_MODULES]

# Passive joints (caster steering + caster rolling + rocker arm) -- not servoed, left free
passive_joint_names = [
    "caster_front_left_steering_joint", "caster_front_left_joint",
    "caster_rear_right_steering_joint", "caster_rear_right_joint",
    "rocker_arm_joint",
]
passive_dofs = [dof(n) for n in passive_joint_names]

# ---- control gains ----
# Arms: very high stiffness/damping to lock the arms firmly in the initial pose
robot.set_dofs_kp(np.array([5000.0] * len(arm_dofs)), arm_dofs)
robot.set_dofs_kv(np.array([500.0] * len(arm_dofs)), arm_dofs)
robot.set_dofs_force_range(np.array([-200.0] * len(arm_dofs)), np.array([200.0] * len(arm_dofs)), arm_dofs)
# Grippers: position control
robot.set_dofs_kp(np.array([200.0] * len(finger_dofs)), finger_dofs)
robot.set_dofs_kv(np.array([20.0] * len(finger_dofs)), finger_dofs)
robot.set_dofs_force_range(np.array([-50.0] * len(finger_dofs)), np.array([50.0] * len(finger_dofs)), finger_dofs)
# Steering joints: position control, high stiffness to hold the steer angle
robot.set_dofs_kp(np.array([500.0] * len(steering_dofs)), steering_dofs)
robot.set_dofs_kv(np.array([50.0] * len(steering_dofs)), steering_dofs)
robot.set_dofs_force_range(np.array([-200.0] * len(steering_dofs)), np.array([200.0] * len(steering_dofs)), steering_dofs)
# Drive wheels: velocity control (kp=0, track target wheel speed via kv)
robot.set_dofs_kp(np.array([0.0] * len(drive_dofs)), drive_dofs)
robot.set_dofs_kv(np.array([50.0] * len(drive_dofs)), drive_dofs)
robot.set_dofs_force_range(np.array([-500.0] * len(drive_dofs)), np.array([500.0] * len(drive_dofs)), drive_dofs)
# Passive joints: zero gains, spin freely
robot.set_dofs_kp(np.array([0.0] * len(passive_dofs)), passive_dofs)
robot.set_dofs_kv(np.array([0.0] * len(passive_dofs)), passive_dofs)

# ---- initial joint pose (arms tucked in, same as the original script's initial_joint_pos) ----
ARM_HOLD = {
    "left_fr3v2_joint1": 0.0, "left_fr3v2_joint2": -1.5, "left_fr3v2_joint3": 0.0,
    "left_fr3v2_joint4": -2.2, "left_fr3v2_joint5": 0.0, "left_fr3v2_joint6": 1.5,
    "left_fr3v2_joint7": 0.785,
    "right_fr3v2_joint1": 0.0, "right_fr3v2_joint2": -1.5, "right_fr3v2_joint3": 0.0,
    "right_fr3v2_joint4": -2.2, "right_fr3v2_joint5": 0.0, "right_fr3v2_joint6": 1.5,
    "right_fr3v2_joint7": 0.785,
}
arm_hold_targets = np.array([ARM_HOLD[n] for n in arm_joint_names])
finger_open = np.array([0.04] * len(finger_dofs))

q_init = robot.get_dofs_position().cpu().numpy()
for n, v in ARM_HOLD.items():
    q_init[dof(n)] = v
for d in finger_dofs:
    q_init[d] = 0.04
robot.set_dofs_position(q_init)

# Let the robot settle on the ground (the base is free, so it settles first)
for _ in range(200):
    robot.control_dofs_position(arm_hold_targets, arm_dofs)
    robot.control_dofs_position(finger_open, finger_dofs)
    scene.step()


########################## base kinematics (numpy version) ##########################
def _wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def _steering_alignment_scale(error):
    """Fade the wheel speed down while the steering is not yet aligned, to avoid side-slip."""
    scale = (STEERING_ZERO_SPEED_ERROR_RAD - error) / (
        STEERING_ZERO_SPEED_ERROR_RAD - STEERING_FULL_SPEED_ERROR_RAD
    )
    return max(0.0, min(1.0, scale))


def get_keyboard_twist(pressed):
    """Keys -> body-frame (vx, vy, wz).

    Note: WASD clashes with Genesis viewer shortcuts, so translation uses the arrow
    keys and rotation uses Q/E (q/e don't clash with the viewer).
        ↑/↓  forward/back      ←/→  strafe left/right      Q/E  rotate in place left/right
    """
    vx = vy = wz = 0.0
    if "up" in pressed:
        vx += LINEAR_SPEED_MPS
    if "down" in pressed:
        vx -= LINEAR_SPEED_MPS
    if "left" in pressed:
        vy += LINEAR_SPEED_MPS
    if "right" in pressed:
        vy -= LINEAR_SPEED_MPS
    if "q" in pressed:
        wz += ANGULAR_SPEED_RADPS
    if "e" in pressed:
        wz -= ANGULAR_SPEED_RADPS
    return vx, vy, wz


def compute_drive_targets(cur_steer_angles, vx, vy, wz):
    """Body twist -> (steer-angle targets, wheel-speed targets), with 180° flip optimization and speed limiting."""
    steer_targets = np.zeros(len(DRIVE_MODULES))
    drive_targets = np.zeros(len(DRIVE_MODULES))

    vectors = []
    max_speed = 0.0
    for (_s, _d, x, y) in DRIVE_MODULES:
        wvx = vx - wz * y
        wvy = vy + wz * x
        sp = math.hypot(wvx, wvy)
        vectors.append((wvx, wvy, sp))
        max_speed = max(max_speed, sp)

    allowed = MAX_WHEEL_SPEED_RADPS * WHEEL_RADIUS_M
    scale = allowed / max_speed if max_speed > allowed else 1.0

    for i, (wvx, wvy, sp) in enumerate(vectors):
        wvx *= scale
        wvy *= scale
        sp *= scale
        cur = cur_steer_angles[i]
        if sp < STOP_EPS:
            steer_targets[i] = cur  # when stopped, hold the current steer angle; don't snap back
            continue
        raw = math.atan2(wvy, wvx)
        direct = _wrap_to_pi(raw - cur)
        flipped = _wrap_to_pi(raw + math.pi - cur)
        use_flipped = abs(flipped) < abs(direct)
        delta = flipped if use_flipped else direct
        steer_targets[i] = cur + delta
        wheel_speed = (sp / WHEEL_RADIUS_M) * _steering_alignment_scale(abs(delta))
        drive_targets[i] = -wheel_speed if use_flipped else wheel_speed

    return steer_targets, drive_targets


def get_root_yaw():
    q = robot.get_quat()
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def get_root_yaw_rate():
    return float(robot.get_ang()[2])


def compensate_yaw_rate(vx, vy, wz, desired_yaw, manual_rotation):
    """Lock the heading during pure translation; reset the held heading on manual rotation / standstill."""
    cur = get_root_yaw()
    if manual_rotation or math.hypot(vx, vy) < STOP_EPS:
        return wz, cur
    err = _wrap_to_pi(desired_yaw - cur)
    comp = HEADING_HOLD_KP * err - HEADING_HOLD_KD * get_root_yaw_rate()
    comp = max(-MAX_HEADING_COMP_RADPS, min(MAX_HEADING_COMP_RADPS, comp))
    return wz + comp, desired_yaw


########################## keyboard listener (pynput, global) ##########################
_pressed = set()
_listener = None


def _on_press(key):
    try:
        _pressed.add(key.char.lower())
    except AttributeError:
        _pressed.add(key.name)


def _on_release(key):
    try:
        _pressed.discard(key.char.lower())
    except AttributeError:
        _pressed.discard(key.name)
    if key == keyboard.Key.esc:
        return False


_listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
_listener.daemon = True
_listener.start()

print("=" * 80)
print("✓ Simulation started!  (Genesis full scene + keyboard control)")
print("  - 11 tables + 9 letters + 3 cutlery items + 1 controllable robot (top center)")
print("=" * 80)
print("Controls: ↑/↓ forward/back | ←/→ strafe | Q/E rotate in place | combinable | ESC to quit")
print("=" * 80)

########################## control loop ##########################
heading_hold_yaw = get_root_yaw()
count = 0
try:
    while True:
        vx, vy, wz_cmd = get_keyboard_twist(_pressed)
        wz, heading_hold_yaw = compensate_yaw_rate(
            vx, vy, wz_cmd, heading_hold_yaw, manual_rotation=abs(wz_cmd) > 1.0e-4
        )

        # Arms + grippers: locked in the initial pose
        robot.control_dofs_position(arm_hold_targets, arm_dofs)
        robot.control_dofs_position(finger_open, finger_dofs)

        # Base: position control for steering, velocity control for the drive wheels
        cur_steer = robot.get_dofs_position(steering_dofs).cpu().numpy()
        steer_targets, drive_targets = compute_drive_targets(cur_steer, vx, vy, wz)
        robot.control_dofs_position(steer_targets, steering_dofs)
        robot.control_dofs_velocity(drive_targets, drive_dofs)

        scene.step()

        count += 1
        if count % 100 == 0 and (vx or vy or wz):
            pos = robot.get_pos().cpu().numpy()
            print(
                f"step {count} | vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} | "
                f"pos [{pos[0]:.2f}, {pos[1]:.2f}] heading {math.degrees(get_root_yaw()):.1f}°"
            )
except KeyboardInterrupt:
    print("\n✓ Stopped by user")
finally:
    if _listener is not None:
        _listener.stop()
    print("✓ Done!")
