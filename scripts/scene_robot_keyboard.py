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
keys and rotation uses , / .):
  ↑/↓ forward/back, ←/→ strafe left/right, , / . rotate in place left/right,
  combinable (e.g. ↑+←), ESC to quit.

Headless: if the DISPLAY env var is empty/unset, the script runs headless — it
opens no viewer and reads no keyboard, instead driving the base through a scripted
path and saving the result to videos/scene_robot_keyboard_<timestamp>.mp4.

Dependency: pip install pynput  (global keyboard listener, same as the original script)

Coordinate / quaternion convention: both Genesis and Isaac Lab use Z-up, meters,
quaternion w-x-y-z, so poses are carried over verbatim.
"""

import math
import os
from datetime import datetime
import warnings

os.environ["TI_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")

# Headless when there is no display (DISPLAY unset or empty). This MUST be decided
# before importing pynput below: pynput opens an X connection at import time and
# crashes on a headless machine. In headless mode we can't open the viewer or read
# the keyboard, so we drive a scripted motion sequence and record it to a video.
# HEADLESS = not os.environ.get("DISPLAY")
HEADLESS = True

import numpy as np
import genesis as gs

# pynput requires a display; import it only in interactive mode so headless runs
# don't fail at import time.
if not HEADLESS:
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
# Assets live one level up, in <repo>/assets/ (moved out of scripts/).
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "assets")
MJCF_DIR = os.path.join(ASSETS_DIR, "mjcf")
URDF_PATH = os.path.join(ASSETS_DIR, "urdf", "mobile_fr3_duo_v0_2_franka_hand.urdf")

def select_backend():
    """Pick a Genesis backend automatically: CUDA -> AMD (ROCm) -> Apple Metal -> CPU.

    Detection is done via PyTorch (a Genesis dependency):
      * a ROCm/HIP build reports torch.version.hip set (and cuda.is_available() True)
      * a CUDA build reports torch.version.cuda set
      * Apple Silicon reports torch.backends.mps available
    Falls back to CPU if torch is missing or no accelerator is found.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return gs.amdgpu if getattr(torch.version, "hip", None) else gs.cuda
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return gs.metal
    except Exception:
        pass
    return gs.cpu


# Auto-select the backend; override by hardcoding e.g. backend=gs.cpu below.
_backend = select_backend()
print(f"[backend] auto-selected: {getattr(_backend, 'name', _backend)}")
gs.init(backend=_backend, theme="light")

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
    show_viewer=not HEADLESS,
)

# In headless mode, add an offscreen camera (same top-down view as the viewer)
# to record the scripted run to videos/.
cam = None
VIDEO_PATH = None
if HEADLESS:
    cam = scene.add_camera(
        res=(960, 640),
        pos=(0.0, -7.0, 14.0),
        lookat=(0.0, 0.0, 0.0),
        fov=40,
        GUI=False,
    )
    _stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    VIDEO_PATH = os.path.join(SCRIPT_DIR, "videos", f"scene_robot_keyboard_{_stamp}.mp4")

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
table_mjcf = os.path.join(MJCF_DIR, "table_edit", "table_edit.xml")

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
    keys and rotation uses , / . (comma/period don't clash with the viewer).
        ↑/↓  forward/back      ←/→  strafe left/right      , / .  rotate in place left/right
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
    if "," in pressed:
        wz += ANGULAR_SPEED_RADPS
    if "." in pressed:
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


########################## headless scripted motion ##########################
# When headless there's no keyboard, so drive the base through a fixed sequence of
# (vx, vy, wz, n_steps) segments to show it off in the recorded video (~7.5 s sim).
HEADLESS_SCRIPT = [
    (0.0, 0.0, 0.0, 100),                   # settle in place
    (LINEAR_SPEED_MPS, 0.0, 0.0, 400),      # drive forward
    (0.0, 0.0, ANGULAR_SPEED_RADPS, 200),   # rotate in place (left)
    (0.0, LINEAR_SPEED_MPS, 0.0, 300),      # strafe left
    (0.0, 0.0, -ANGULAR_SPEED_RADPS, 200),  # rotate in place (right)
    (-LINEAR_SPEED_MPS, 0.0, 0.0, 300),     # reverse
]
HEADLESS_TOTAL_STEPS = sum(seg[3] for seg in HEADLESS_SCRIPT)
RENDER_EVERY = 4  # render every 4th sim step -> ~50 fps at dt=0.005 (≈ real time)


def scripted_twist(step):
    """Body-frame (vx, vy, wz) for the given headless step index from HEADLESS_SCRIPT."""
    t = 0
    for vx, vy, wz, n in HEADLESS_SCRIPT:
        if step < t + n:
            return vx, vy, wz
        t += n
    return 0.0, 0.0, 0.0


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


# Only listen for keys in interactive mode; headless has no display/keyboard.
if not HEADLESS:
    _listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    _listener.daemon = True
    _listener.start()

print("=" * 80)
if HEADLESS:
    print("✓ Simulation started!  (Genesis full scene -- HEADLESS, recording video)")
    print("  - 11 tables + 9 letters + 3 cutlery items + 1 robot driving a scripted path")
    print(f"  - DISPLAY is empty -> headless; video will be saved to {VIDEO_PATH}")
else:
    print("✓ Simulation started!  (Genesis full scene + keyboard control)")
    print("  - 11 tables + 9 letters + 3 cutlery items + 1 controllable robot (top center)")
    print("  Controls: ↑/↓ forward/back | ←/→ strafe | , / . rotate in place | combinable | ESC to quit")
print("=" * 80)


def drive_step(vx, vy, wz_cmd, heading_hold_yaw):
    """Apply one control step (arms locked, base driven) and advance the sim by one step."""
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
    return wz, heading_hold_yaw


def log_progress(count, vx, vy, wz):
    if count % 100 == 0 and (vx or vy or wz):
        pos = robot.get_pos().cpu().numpy()
        print(
            f"step {count} | vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} | "
            f"pos [{pos[0]:.2f}, {pos[1]:.2f}] heading {math.degrees(get_root_yaw()):.1f}°"
        )


########################## control loop ##########################
heading_hold_yaw = get_root_yaw()

if HEADLESS:
    # Run the scripted path for a fixed number of steps, recording every RENDER_EVERY-th frame.
    cam.start_recording()
    for count in range(HEADLESS_TOTAL_STEPS):
        vx, vy, wz_cmd = scripted_twist(count)
        wz, heading_hold_yaw = drive_step(vx, vy, wz_cmd, heading_hold_yaw)
        if count % RENDER_EVERY == 0:
            cam.render()
        log_progress(count, vx, vy, wz)
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    cam.stop_recording(save_to_filename=VIDEO_PATH, fps=50)
    print(f"✓ Saved video to {VIDEO_PATH}")
else:
    count = 0
    try:
        while True:
            vx, vy, wz_cmd = get_keyboard_twist(_pressed)
            wz, heading_hold_yaw = drive_step(vx, vy, wz_cmd, heading_hold_yaw)
            count += 1
            log_progress(count, vx, vy, wz)
    except KeyboardInterrupt:
        print("\n✓ Stopped by user")
    finally:
        if _listener is not None:
            _listener.stop()

print("✓ Done!")
