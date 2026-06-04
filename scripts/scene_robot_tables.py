#!/usr/bin/env python3
"""
Reusable robot scene (Genesis): 11 tables + 9 letters + 3 cutlery items + 1 mobile dual-arm robot.

it is now a small library exposing a single ``RobotScene`` class that can be either:

  * imported into a notebook / another script::

        from scene_robot_tables import RobotScene
        sim = RobotScene(headless=True)
        sim.set_base_velocity(0.5, 0.0, 0.0, steps=200)   # drive the base forward
        sim.move_ee("left", (0.4, 4.9, 0.9))         # IK-move the left end-effector
        frame = sim.render()                         # grab an RGB frame

  * run from the command line::

        python scene_robot_tables.py --headless --save-video

Importing this module has **no side effects** (it does not call ``gs.init`` or build a
scene); all Genesis setup happens inside ``RobotScene.__init__``.

``headless`` controls only whether the interactive viewer window opens. When ``None`` it
is auto-detected from the ``DISPLAY`` env var (empty/unset -> headless).

Recording is opt-in and independent of ``headless``: an offscreen camera is always created
(so ``render()`` for inline notebook frames works regardless), but mp4 capture happens only
if the caller calls ``start_recording()`` (then ``save_video(path)`` to write it).

Coordinate / quaternion convention: Genesis is Z-up, meters, quaternion w-x-y-z.
"""

import argparse
import logging
import math
import os
import shutil
import warnings
from datetime import datetime

import numpy as np
import genesis as gs

os.environ.setdefault("TI_LOG_LEVEL", "error")
warnings.filterwarnings("ignore")

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

# Passive joints (caster steering + caster rolling + rocker arm) -- not servoed, left free.
PASSIVE_JOINT_NAMES = [
    "caster_front_left_steering_joint", "caster_front_left_joint",
    "caster_rear_right_steering_joint", "caster_rear_right_joint",
    "rocker_arm_joint",
]

# Initial tucked arm pose (same as the original script's initial_joint_pos).
ARM_HOLD = {
    "left_fr3v2_joint1": 0.0, "left_fr3v2_joint2": -1.5, "left_fr3v2_joint3": 0.0,
    "left_fr3v2_joint4": -2.2, "left_fr3v2_joint5": 0.0, "left_fr3v2_joint6": 1.5,
    "left_fr3v2_joint7": 0.785,
    "right_fr3v2_joint1": 0.0, "right_fr3v2_joint2": -1.5, "right_fr3v2_joint3": 0.0,
    "right_fr3v2_joint4": -2.2, "right_fr3v2_joint5": 0.0, "right_fr3v2_joint6": 1.5,
    "right_fr3v2_joint7": 0.785,
}

GRIPPER_OPEN = 0.04  # finger opening, meters (per finger)

# Vertical spine lift (prismatic) that raises/lowers the whole dual-arm mount.
# URDF limits: 0.0 (lowest) .. 0.85 m, effort 100 N. It must actively hold the arm
# assembly against gravity, otherwise the DOF (kp=0 by default) sags. We position-
# servo it and hold it at SPINE_HOLD by default.
SPINE_JOINT_NAME = "franka_spine_vertical_joint"
SPINE_LOWER, SPINE_UPPER = 0.0, 0.85
SPINE_HOLD = 0.0  # default lift height, meters


def _quat_mul(a, b):
    """Hamilton product of two w-x-y-z quaternions (a applied first, then b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


# The Franka hand frame sits 45° about z from link7 (where we run IK), so a plain
# "straight down" target [0,1,0,0] leaves the gripper yawed 45° in the world. Fold a
# -45° z-rotation into the target to cancel it. Flip GRIPPER_YAW_FIX's sign if the
# fingers end up rotated the *other* way by 90°.
GRIPPER_YAW_FIX = -math.pi / 4
_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # link7 pointing straight down (w-x-y-z)
_RZ = np.array([math.cos(0.5 * GRIPPER_YAW_FIX), 0.0, 0.0, math.sin(0.5 * GRIPPER_YAW_FIX)])
DOWN_QUAT = _quat_mul(_DOWN, _RZ)  # gripper down, fingers aligned to world axes
# Genesis fuses the fixed-joint hand_tcp frame into link7, so we IK on link7 and
# offset down to the finger TCP 
TCP_OFFSET = 0.166

RENDER_EVERY = 8   # record every 4th sim step -> ~50 fps at dt=0.005 (≈ real time)
VIDEO_FPS = 25

# Robot-mounted head camera (attached to the URDF's head_camera_mounting_point link).
HEAD_CAM_LINK = "head_camera_mounting_point"
HEAD_CAM_RES = (960, 640)
HEAD_CAM_FOV = 90
# Camera pose relative to the mount link. The link's local +X already points forward and
# ~41° down (the mount joint is pitched about Y), and Genesis cameras look down their local
# -Z with +Y up, so this rotation maps: cam -Z -> link +X (view forward/down), cam +Y ->
# link +Z (up). Columns are the camera axes expressed in the link frame.
HEAD_CAM_OFFSET_T = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

########################## asset paths ##########################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_assets_dir():
    """Locate the assets/ directory in both the source repo and an installed wheel.

    * Installed as the ``fr3_genesis`` package, assets are bundled alongside this
      module: ``<package>/assets``.
    * Running from the source tree (scripts/scene_robot_tables.py), they live one
      level up at ``<repo>/assets``.
    """
    for candidate in (
        os.path.join(SCRIPT_DIR, "assets"),                  # installed wheel
        os.path.join(os.path.dirname(SCRIPT_DIR), "assets"),  # source repo
    ):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(os.path.dirname(SCRIPT_DIR), "assets")


ASSETS_DIR = _find_assets_dir()
MJCF_DIR = os.path.join(ASSETS_DIR, "mjcf")
URDF_PATH = os.path.join(ASSETS_DIR, "urdf", "mobile_fr3_duo_v0_2_franka_hand.urdf")
VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")

# Genesis convex-decomposition cache. The first build of this scene runs (slow) convex
# decomposition on every collision mesh and writes the result to ~/.cache/genesis. We
# ship a pre-computed copy under assets/genesis_cache so a fresh machine (or container)
# can skip that step. On startup, if the user's cache is missing/empty we seed it from
# the bundled copy; if it already exists we leave it untouched.
GENESIS_CACHE_BUNDLED = os.path.join(ASSETS_DIR, "genesis_cache")
GENESIS_CACHE_HOME = os.path.expanduser("~/.cache/genesis")


def _restore_genesis_cache():
    """Seed ~/.cache/genesis from the bundled assets cache if it isn't already populated."""
    bundled_cvx = os.path.join(GENESIS_CACHE_BUNDLED, "cvx")
    if not os.path.isdir(bundled_cvx):
        return  # nothing bundled to restore

    home_cvx = os.path.join(GENESIS_CACHE_HOME, "cvx")
    if os.path.isdir(home_cvx) and os.listdir(home_cvx):
        return  # already cached -> leave it alone

    os.makedirs(GENESIS_CACHE_HOME, exist_ok=True)
    shutil.copytree(bundled_cvx, home_cvx, dirs_exist_ok=True)
    print(f"[cache] seeded genesis convex-decomposition cache -> {home_cvx}")


########################## backend / init helpers ##########################
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


# gs.init is once-per-process; guard it so re-instantiating RobotScene in a notebook
# (e.g. re-running a cell) does not crash with "already initialized".
_GS_INITIALIZED = False


def _ensure_gs_init(backend):
    global _GS_INITIALIZED
    if _GS_INITIALIZED:
        return
    _restore_genesis_cache()
    if backend is None:
        backend = select_backend()
    # ROCm/HIP needs 64-bit precision to stay numerically stable; others use 32-bit.
    precision = "64" if backend is gs.amdgpu else "32"
    print(f"[backend] using: {getattr(backend, 'name', backend)} (precision={precision})")
    gs.init(backend=backend, theme="light", precision=precision)
    gs.logger._logger.setLevel(logging.WARNING)
    _GS_INITIALIZED = True


########################## small numpy kinematics helpers ##########################
def _wrap_to_pi(a):
    """Wrap angle(s) to (-pi, pi]. Accepts a scalar or a numpy array."""
    return np.arctan2(np.sin(a), np.cos(a))


def _steering_alignment_scale(error):
    """Fade wheel speed down while the steering is not yet aligned, to avoid side-slip.

    Accepts a scalar or a numpy array; returns the same shape, clamped to [0, 1].
    """
    scale = (STEERING_ZERO_SPEED_ERROR_RAD - error) / (
        STEERING_ZERO_SPEED_ERROR_RAD - STEERING_FULL_SPEED_ERROR_RAD
    )
    return np.clip(scale, 0.0, 1.0)


def quat_to_yaw(quat):
    """Yaw (rad) about world z from a (w, x, y, z) quaternion. Accepts [4] or [B, 4]."""
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# Module geometry, cached as [M] arrays for the vectorized drive solver.
_MODULE_X = np.array([m[2] for m in DRIVE_MODULES], dtype=float)
_MODULE_Y = np.array([m[3] for m in DRIVE_MODULES], dtype=float)


def compute_drive_targets(cur_steer_angles, vx, vy, wz):
    """Body twist -> (steer-angle targets, wheel-speed targets), batched over envs.

    Parameters
    ----------
    cur_steer_angles : array [B, M]   current steer angle per module, per env
    vx, vy, wz        : array [B]      body-frame twist per env
    Returns (steer_targets[B, M], drive_targets[B, M]).

    Vectorized form of the original per-module scalar loop: per-env global speed
    limiting, the 180-degree steer-flip optimization, holding the steer angle when
    a module's commanded speed is ~0, and the steering-alignment speed fade.
    """
    cur = np.atleast_2d(np.asarray(cur_steer_angles, dtype=float))  # [B, M]
    vx = np.atleast_1d(np.asarray(vx, dtype=float))                 # [B]
    vy = np.atleast_1d(np.asarray(vy, dtype=float))
    wz = np.atleast_1d(np.asarray(wz, dtype=float))

    # Per-module world velocity at each wheel: [B, M]
    wvx = vx[:, None] - wz[:, None] * _MODULE_Y[None, :]
    wvy = vy[:, None] + wz[:, None] * _MODULE_X[None, :]
    sp = np.hypot(wvx, wvy)                                         # [B, M]

    # Per-env speed limit, scaling the whole env's module set together.
    allowed = MAX_WHEEL_SPEED_RADPS * WHEEL_RADIUS_M
    max_speed = sp.max(axis=1)                                     # [B]
    scale = np.where(max_speed > allowed, allowed / np.maximum(max_speed, 1e-12), 1.0)
    scale = scale[:, None]                                         # [B, 1]
    wvx = wvx * scale
    wvy = wvy * scale
    sp = sp * scale

    moving = sp >= STOP_EPS                                        # [B, M]
    raw = np.arctan2(wvy, wvx)
    direct = _wrap_to_pi(raw - cur)
    flipped = _wrap_to_pi(raw + math.pi - cur)
    use_flipped = np.abs(flipped) < np.abs(direct)
    delta = np.where(use_flipped, flipped, direct)

    # Hold current steer where stopped; otherwise rotate by the smaller delta.
    steer_targets = np.where(moving, cur + delta, cur)

    wheel_speed = (sp / WHEEL_RADIUS_M) * _steering_alignment_scale(np.abs(delta))
    drive = np.where(use_flipped, -wheel_speed, wheel_speed)
    drive_targets = np.where(moving, drive, 0.0)

    return steer_targets, drive_targets


def _with_suffix(path, suffix):
    """Insert ``suffix`` before the file extension: foo.mp4 -> foo<suffix>.mp4."""
    root, ext = os.path.splitext(path)
    return root + suffix + ext


########################## the reusable scene ##########################
class RobotScene:
    """A built Genesis scene with a mobile dual-arm robot, ready to drive and control.

    Parameters
    ----------
    headless : bool | None
        If True, no interactive viewer opens. If None (default), auto-detected from the
        DISPLAY env var. Controls only the viewer; an offscreen render camera always exists.
    backend : genesis backend | None
        e.g. ``gs.cpu``. None -> auto (``select_backend``), which honors the
        ``FR3_BACKEND`` env var (cpu/cuda/amd/metal) to force a backend before
        falling back to auto-detection. An explicit value here overrides both.
    camera_res : (int, int)
        Offscreen camera resolution (width, height).
    n_envs : int
        Number of parallel environments. 1 (default) builds a single non-batched
        scene with the classic scalar API. >1 builds ``n_envs`` copies; getters
        then return a leading ``[n_envs, ...]`` dim and setters accept either a
        scalar/per-dof value (broadcast to all envs) or a full ``[n_envs, ...]`` array.
    env_spacing : (float, float)
        Grid spacing between parallel envs (only used when n_envs > 1).

    Recording is opt-in and decided by the caller: call ``start_recording()`` to begin
    capturing frames, then ``save_video(path)`` (or ``close()``) to write the mp4. If you
    never call ``start_recording()``, no frames are captured and ``save_video()`` is a no-op.
    """

    def __init__(self, headless=None, save_video=False, video_path=None,
                 backend=None, camera_res=(960, 640), n_envs=1, env_spacing=(2.0, 2.0)):
        if int(n_envs) < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        if headless is None:
            headless = not os.environ.get("DISPLAY")
        self.headless = headless
        self.n_envs = int(n_envs)
        self.batched = self.n_envs > 1
        self.env_spacing = env_spacing
        self._record = bool(save_video)
        self.video_path = video_path
        self._recording = False
        self._frame_count = 0

        _ensure_gs_init(backend)

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, -7.0, 14.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.005, gravity=(0.0, 0.0, -9.81)),
            show_viewer=not headless,
        )

        # Offscreen camera (always present) -- used for inline frames and optional recording.
        self.cam = self.scene.add_camera(
            res=camera_res,
            pos=(0.0, -7.0, 14.0),
            lookat=(0.0, 0.0, 0.0),
            fov=40,
            GUI=False,
        )

        # Robot-mounted head camera (pose is bound to the head link after build).
        self.head_cam = self.scene.add_camera(
            res=HEAD_CAM_RES,
            fov=HEAD_CAM_FOV,
            GUI=False,
        )

        self._build_scene()

        # if not headless:
        #     self.scene.viewer.add_plugin(
        #         gs.vis.viewer_plugins.MouseInteractionPlugin(
        #             use_force=True,  # False = set position, True = spring force
        #             spring_const=1000.0,
        #             color=(0.1, 0.6, 0.8, 0.6),
        #         )
        #     )
        if self.batched:
            self.scene.build(n_envs=self.n_envs, env_spacing=self.env_spacing)
        else:
            self.scene.build()

        self._setup_handles()
        self._setup_gains()
        self._apply_initial_pose()

        # Bind the head camera to the robot's head-camera mount link.
        self.head_link = self.robot.get_link(HEAD_CAM_LINK)
        self.head_cam.attach(self.head_link, HEAD_CAM_OFFSET_T)
        self.head_cam.move_to_attach()

        # Base heading-hold state (batched [N, ...]), reset after settling.
        self._twist = np.zeros((self.n_envs, 3))
        self._heading_hold_yaw = self._yaw_all()

    # ------------------------------------------------------------------ build
    def _build_scene(self):
        # ---- ground (high friction so the drive wheels grip) ----
        self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=2.0),
        )

        # ---- 11 tables (welded to world) ----
        table_mjcf = os.path.join(MJCF_DIR, "table_edit", "table_edit.xml")
        table_positions = [
            (-2.0, 3.0, 0.0), (-2.0, 1.5, 0.0), (-2.0, 0.0, 0.0), (-2.0, -1.5, 0.0), (-2.0, -3.0, 0.0),
            (2.0, 3.0, 0.0), (2.0, 1.5, 0.0), (2.0, 0.0, 0.0), (2.0, -1.5, 0.0), (2.0, -3.0, 0.0),
            (0.0, -4.5, 0.0),  # bottom center (top center is reserved for the robot)
        ]
        for pos in table_positions:
            self.scene.add_entity(gs.morphs.MJCF(file=table_mjcf, pos=pos))

        # ---- 9 letters (black), each needs +90° about X to stand upright ----
        letter_black = gs.surfaces.Default(color=(0.0, 0.0, 0.0))
        letter_quat = (0.7071, 0.7071, 0.0, 0.0)
        letter_table_pos = {
            "A": (-2.0, 1.5, 0.7), "B": (2.0, 1.5, 0.7),
            "C": (-2.0, 0.0, 0.7), "D": (2.0, 0.0, 0.7),
            "E": (-2.0, -1.5, 0.7), "F": (2.0, -1.5, 0.7),
            "G": (-2.0, -3.0, 0.7), "H": (2.0, -3.0, 0.7),
            "I": (0.0, -4.5, 0.7),
        }
        for letter, (tx, ty, tz) in letter_table_pos.items():
            letter_mjcf = os.path.join(MJCF_DIR, f"{letter}_edit", f"{letter}_edit.xml")
            self.scene.add_entity(
                gs.morphs.MJCF(file=letter_mjcf, pos=(tx, ty, tz + 0.061), quat=letter_quat),
                surface=letter_black,
            )

        # ---- 3 cutlery items on the top-left table ----
        ikea_table_pos = (-2.0, 3.0, 0.7)
        cutlery_configs = {
            "bowl": {"offset": (0.12, 0.0, 0.15), "color": (1.0, 0.0, 0.0)},
            "plate": {"offset": (-0.12, 0.0, 0.13), "color": (1.0, 1.0, 0.0)},
            "spoon": {"offset": (0.0, 0.15, 0.13), "color": (0.0, 0.0, 1.0)},
        }
        for item, cfg in cutlery_configs.items():
            ox, oy, oz = cfg["offset"]
            item_pos = (ikea_table_pos[0] + ox, ikea_table_pos[1] + oy, ikea_table_pos[2] + oz)
            item_mjcf = os.path.join(MJCF_DIR, item, f"{item}.xml")
            self.scene.add_entity(
                gs.morphs.MJCF(file=item_mjcf, pos=item_pos),
                surface=gs.surfaces.Default(color=cfg["color"]),
            )

        # ---- a small pickable cube on the cutlery table (robot-facing corner) ----
        # Dynamic (not fixed) so an arm can grasp it; it settles onto the table during
        # the init settle loop. Exposed as ``self.cube`` so callers can read its live
        # pose with ``self.cube.get_pos()``.
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(-1.82, 3.18, 0.80)),
            surface=gs.surfaces.Default(color=(0.0, 0.8, 0.2)),
            material=gs.materials.Rigid(friction=2.0),
        )

        # ---- robot (free base, top center) ----
        # links_to_keep preserves the head-camera mount link (a fixed-joint link that would
        # otherwise be merged away) so the head camera can be attached to it after build.
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=URDF_PATH, pos=(0.0, 4.5, 0.0), fixed=False,
                links_to_keep=(HEAD_CAM_LINK,),
            ),
            material=gs.materials.Rigid(friction=2.0),
        )

    # ------------------------------------------------------------- handles
    def _dof(self, name):
        return self.robot.get_joint(name).dofs_idx_local[0]

    def _setup_handles(self):
        # Arms (7 per side) and grippers (2 fingers per side), looked up by name
        # because L/R dof indices are interleaved.
        self.arm_joint_names = {
            side: [f"{side}_fr3v2_joint{i}" for i in range(1, 8)] for side in ("left", "right")
        }
        self.finger_joint_names = {
            side: [f"{side}_fr3v2_finger_joint{j}" for j in (1, 2)] for side in ("left", "right")
        }
        self.arm_dofs = {s: [self._dof(n) for n in self.arm_joint_names[s]] for s in ("left", "right")}
        self.finger_dofs = {s: [self._dof(n) for n in self.finger_joint_names[s]] for s in ("left", "right")}
        # qpos indices for the arm joints. With a free base, qpos (n_qs) and dof (n_dofs)
        # indices differ by 1, and inverse_kinematics returns a qpos-indexed array, so the
        # IK result must be read with these, not with dofs_idx_local.
        self.arm_qs = {
            s: [self.robot.get_joint(n).qs_idx_local[0] for n in self.arm_joint_names[s]]
            for s in ("left", "right")
        }
        # IK target frame: link7 (Genesis fuses the hand_tcp fixed link into it).
        self.ee_link = {s: self.robot.get_link(f"{s}_fr3v2_link7") for s in ("left", "right")}

        # Vertical spine lift (single prismatic dof), position-servoed and held.
        self.spine_dof = self._dof(SPINE_JOINT_NAME)

        # Active base joints.
        self.steering_dofs = [self._dof(m[0]) for m in DRIVE_MODULES]
        self.drive_dofs = [self._dof(m[1]) for m in DRIVE_MODULES]
        self.passive_dofs = [self._dof(n) for n in PASSIVE_JOINT_NAMES]

        # Convenience: all arm / finger dofs in a fixed order.
        self._all_arm_dofs = self.arm_dofs["left"] + self.arm_dofs["right"]
        self._all_finger_dofs = self.finger_dofs["left"] + self.finger_dofs["right"]

        # Current control targets (updated by set_arm / set_gripper, re-applied each
        # step). Stored batched as [N, width] so each env can hold a different target.
        self._arm_target = {
            s: np.tile(np.array([ARM_HOLD[n] for n in self.arm_joint_names[s]]), (self.n_envs, 1))
            for s in ("left", "right")
        }
        self._finger_target = {s: np.full((self.n_envs, 2), GRIPPER_OPEN) for s in ("left", "right")}
        self._spine_target = np.full(self.n_envs, SPINE_HOLD)

    def _setup_gains(self):
        r = self.robot
        n_arm = len(self._all_arm_dofs)
        n_fin = len(self._all_finger_dofs)
        n_steer = len(self.steering_dofs)
        n_drive = len(self.drive_dofs)
        n_pass = len(self.passive_dofs)
        # Arms: stiff position control.
        r.set_dofs_kp(np.full(n_arm, 5000.0), self._all_arm_dofs)
        r.set_dofs_kv(np.full(n_arm, 500.0), self._all_arm_dofs)
        r.set_dofs_force_range(np.full(n_arm, -200.0), np.full(n_arm, 200.0), self._all_arm_dofs)
        # Vertical spine lift: stiff position control (must hold the arm assembly's
        # weight against gravity; generous force range so it doesn't sag).
        r.set_dofs_kp(np.array([8000.0]), [self.spine_dof])
        r.set_dofs_kv(np.array([800.0]), [self.spine_dof])
        r.set_dofs_force_range(np.array([-2000.0]), np.array([2000.0]), [self.spine_dof])
        # Grippers: position control.
        r.set_dofs_kp(np.full(n_fin, 200.0), self._all_finger_dofs)
        r.set_dofs_kv(np.full(n_fin, 20.0), self._all_finger_dofs)
        r.set_dofs_force_range(np.full(n_fin, -50.0), np.full(n_fin, 50.0), self._all_finger_dofs)
        # Steering: stiff position control.
        r.set_dofs_kp(np.full(n_steer, 500.0), self.steering_dofs)
        r.set_dofs_kv(np.full(n_steer, 50.0), self.steering_dofs)
        r.set_dofs_force_range(np.full(n_steer, -200.0), np.full(n_steer, 200.0), self.steering_dofs)
        # Drive wheels: velocity control (kp=0).
        r.set_dofs_kp(np.full(n_drive, 0.0), self.drive_dofs)
        r.set_dofs_kv(np.full(n_drive, 50.0), self.drive_dofs)
        r.set_dofs_force_range(np.full(n_drive, -500.0), np.full(n_drive, 500.0), self.drive_dofs)
        # Passive joints: free.
        r.set_dofs_kp(np.full(n_pass, 0.0), self.passive_dofs)
        r.set_dofs_kv(np.full(n_pass, 0.0), self.passive_dofs)

    # --- batch I/O adapters: the controller core always sees [N, ...] ----
    def _read(self, tensor):
        """numpy view of a Genesis state tensor, always shaped [N, ...]."""
        arr = tensor.cpu().numpy()
        return arr if self.batched else arr[None, ...]

    def _read_dofs(self, dofs):
        return self._read(self.robot.get_dofs_position(dofs))     # [N, len(dofs)]

    def _emit(self, arr):
        """Squeeze the batch dim back out when non-batched, for Genesis writes."""
        return arr if self.batched else arr[0]

    def _broadcast(self, value, width):
        """Normalize a setter argument to [N, width].

        Accepts a scalar, a [width] per-dof vector (broadcast to all envs), or a
        full [N, width] array. Raises ValueError on any other shape.
        """
        a = np.asarray(value, dtype=float)
        if a.ndim == 0:
            return np.full((self.n_envs, width), float(a))
        if a.ndim == 1 and a.shape[0] == width:
            return np.tile(a, (self.n_envs, 1))
        if a.ndim == 2 and a.shape == (self.n_envs, width):
            return a.astype(float)
        raise ValueError(
            f"expected scalar, [{width}], or [{self.n_envs}, {width}]; got shape {a.shape}")

    def _yaw_all(self):
        """Yaw of every env as [N] (uses the batched quaternion read)."""
        return quat_to_yaw(self._read(self.robot.get_quat()))

    def _yaw_rate_all(self):
        """Yaw rate (about z) of every env as [N]."""
        return self._read(self.robot.get_ang())[:, 2]

    def _apply_initial_pose(self):
        q_init = self._read(self.robot.get_dofs_position())       # [N, n_dofs]
        # Set the held joints on every env.
        for n, v in ARM_HOLD.items():
            q_init[:, self._dof(n)] = v
        for d in self._all_finger_dofs:
            q_init[:, d] = GRIPPER_OPEN
        q_init[:, self.spine_dof] = SPINE_HOLD
        self.robot.set_dofs_position(self._emit(q_init))
        # Let the free base settle on the ground.
        for _ in range(200):
            self._apply_joint_targets()
            self.scene.step()

    # ----------------------------------------------------------- control
    def _apply_joint_targets(self):
        """Re-issue the stored arm + gripper + spine position targets (every sim step)."""
        self.robot.control_dofs_position(self._emit(self._spine_target[:, None]), [self.spine_dof])
        for s in ("left", "right"):
            self.robot.control_dofs_position(self._emit(self._arm_target[s]), self.arm_dofs[s])
            self.robot.control_dofs_position(self._emit(self._finger_target[s]), self.finger_dofs[s])

    def _apply_base_control(self):
        """Recompute and issue steer/drive commands from the stored twist + heading hold."""
        vx = self._twist[:, 0]
        vy = self._twist[:, 1]
        wz_cmd = self._twist[:, 2]
        cur_yaw = self._yaw_all()                                   # [N]

        # Per env: hold heading unless rotating or (near-)stationary.
        manual_rotation = np.abs(wz_cmd) > STOP_EPS
        translating = np.hypot(vx, vy) >= STOP_EPS
        hold = (~manual_rotation) & translating                    # [N] bool

        err = _wrap_to_pi(self._heading_hold_yaw - cur_yaw)
        comp = HEADING_HOLD_KP * err - HEADING_HOLD_KD * self._yaw_rate_all()
        comp = np.clip(comp, -MAX_HEADING_COMP_RADPS, MAX_HEADING_COMP_RADPS)
        wz = np.where(hold, wz_cmd + comp, wz_cmd)

        # Re-anchor the heading reference for envs not currently holding.
        self._heading_hold_yaw = np.where(hold, self._heading_hold_yaw, cur_yaw)

        cur_steer = self._read_dofs(self.steering_dofs)            # [N, M]
        steer_targets, drive_targets = compute_drive_targets(cur_steer, vx, vy, wz)
        self.robot.control_dofs_position(self._emit(steer_targets), self.steering_dofs)
        self.robot.control_dofs_velocity(self._emit(drive_targets), self.drive_dofs)

    def step(self, n=1):
        """Advance the simulation by ``n`` steps, holding all current targets."""
        for _ in range(n):
            self._apply_base_control()
            self._apply_joint_targets()
            self.scene.step()
            if self._recording and self._frame_count % RENDER_EVERY == 0:
                self.cam.render(rgb=True)
                self.head_cam.move_to_attach()  # keep the head cam on the moving link
                self.head_cam.render(rgb=True)
            self._frame_count += 1
        return self

    # base ------------------------------------------------------------
    def set_base_velocity(self, vx, vy, wz, steps=1):
        """Set the body-frame base velocity (vx fwd, vy left, wz yaw) and advance ``steps`` steps."""
        self._twist = (float(vx), float(vy), float(wz))
        return self.step(steps)

    def stop_base(self, steps=1):
        """Zero the base velocity (and optionally keep stepping in place)."""
        return self.set_base_velocity(0.0, 0.0, 0.0, steps)

    def reset_pose(self, settle=100):
        """Return both arms to the tucked ``ARM_HOLD`` pose and open the grippers.

        Drives the joints back via the position controllers (then steps ``settle`` times
        to let them arrive) -- useful to undo earlier joint/IK moves before a new task.
        """
        for s in ("left", "right"):
            self._arm_target[s] = np.array([ARM_HOLD[n] for n in self.arm_joint_names[s]])
            self._finger_target[s] = np.full(2, GRIPPER_OPEN)
        self.step(settle)
        return self

    def teleport_base(self, x, y, yaw=None, settle=100):
        """Instantly move the base to world ``(x, y)`` (z kept) with optional ``yaw`` (rad).

        Unlike ``set_base_velocity`` this does not drive the wheels -- it snaps the free
        base to the new pose, zeroes the commanded twist, then steps ``settle`` times so
        the arms re-settle and the heading-hold reference is re-anchored.
        """
        pos = self.get_pos()
        pos[0], pos[1] = float(x), float(y)
        self.robot.set_pos(pos)
        if yaw is not None:
            half = 0.5 * float(yaw)
            self.robot.set_quat(np.array([math.cos(half), 0.0, 0.0, math.sin(half)]))
        self._twist = (0.0, 0.0, 0.0)
        self.step(settle)
        self._heading_hold_yaw = self.get_yaw()
        return self

    def get_pos(self):
        return self.robot.get_pos().cpu().numpy()

    def get_yaw(self):
        q = self.robot.get_quat()
        w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def get_yaw_rate(self):
        return float(self.robot.get_ang()[2])

    def get_base_pose(self):
        """Return ((x, y, z), yaw) of the robot base."""
        return self.get_pos(), self.get_yaw()

    # joints ----------------------------------------------------------
    def set_arm(self, side, q, step=False, steps=1):
        """Set the 7 arm joint position targets for ``side`` ('left'/'right')."""
        self._arm_target[side] = np.asarray(q, dtype=float).reshape(7)
        if step:
            self.step(steps)
        return self

    def get_arm(self, side):
        return self.robot.get_dofs_position(self.arm_dofs[side]).cpu().numpy()

    def set_gripper(self, side, opening, step=False, steps=1):
        """Set gripper opening (meters per finger, 0=closed .. 0.04=open) for ``side``."""
        self._finger_target[side] = np.full(2, float(opening))
        if step:
            self.step(steps)
        return self

    def set_spine(self, height, step=False, steps=1):
        """Set the vertical spine lift height (meters), clamped to [0.0, 0.85]."""
        self._spine_target = max(SPINE_LOWER, min(SPINE_UPPER, float(height)))
        if step:
            self.step(steps)
        return self

    def get_spine(self):
        """Current vertical spine lift height (meters)."""
        return float(self.robot.get_dofs_position([self.spine_dof]).cpu().numpy()[0])

    # IK --------------------------------------------------------------
    def _tcp_of(self, side):
        """Current finger-TCP position of ``side`` (link7 origin minus the tool offset)."""
        link7 = self.ee_link[side].get_pos().cpu().numpy()
        return link7 - np.array([0.0, 0.0, TCP_OFFSET])

    def get_ee_pos(self, side):
        """Current finger tool-center-point (x, y, z) of the ``side`` arm."""
        return self._tcp_of(side)

    def ik(self, side, pos, quat=DOWN_QUAT):
        """Return the 7 arm joint angles placing ``side`` finger TCP at ``pos`` with ``quat``.

        ``pos`` is the finger tool-center-point; it is offset up to the link7 frame that
        Genesis actually exposes (assumes a roughly gripper-down orientation).
        """
        target = np.asarray(pos, dtype=float) + np.array([0.0, 0.0, TCP_OFFSET])
        q_full = self.robot.inverse_kinematics(
            link=self.ee_link[side],
            pos=target,
            quat=np.asarray(quat, dtype=float),
            dofs_idx_local=self.arm_dofs[side],
        )
        # q_full is qpos-indexed; read the arm's qpos slots (offset from dof indices
        # by the free base). The values are joint angles, usable as dof position targets.
        return q_full[self.arm_qs[side]].cpu().numpy()

    def move_ee(self, side, pos, quat=DOWN_QUAT, n_waypoints=60, settle=2):
        """Move ``side`` finger TCP to ``pos`` along an interpolated straight line of way-points."""
        start = self._tcp_of(side)
        target = np.asarray(pos, dtype=float)
        for i in range(1, n_waypoints + 1):
            p = start + (target - start) * i / n_waypoints
            self.set_arm(side, self.ik(side, p, quat))
            self.step(settle)
        return self

    # rendering / video ----------------------------------------------
    def render(self):
        """Return one RGB frame (H, W, 3 uint8) from the top-down offscreen camera."""
        out = self.cam.render()
        rgb = out[0] if isinstance(out, (tuple, list)) else out
        return np.asarray(rgb)

    def render_head(self):
        """Return one RGB frame from the robot-mounted head camera (follows the head link)."""
        self.head_cam.move_to_attach()  # update pose to the current head-link transform
        out = self.head_cam.render()
        rgb = out[0] if isinstance(out, (tuple, list)) else out
        return np.asarray(rgb)

    def start_recording(self):
        """Begin accumulating frames (every RENDER_EVERY-th step) for both cameras.

        Recording is opt-in: call this to enable it. ``save_video()`` then writes the mp4.
        """
        if not self._recording:
            self.cam.start_recording()
            self.head_cam.start_recording()
            self._recording = True
        return self

    def save_video(self, path=None, fps=VIDEO_FPS):
        """Write the recorded frames and stop recording.

        Two files are written, derived from ``path`` (or a timestamped default):
        a ``*_top.mp4`` (top-down camera) and a ``*_head.mp4`` (robot head camera).
        Returns ``(top_path, head_path)``.
        """
        if not self._recording:
            print("[video] nothing recorded (save_video=False / start_recording not called)")
            return None
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(VIDEO_DIR, f"scene_robot_tables_{stamp}.mp4")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        top_path = _with_suffix(path, "_top")
        head_path = _with_suffix(path, "_head")
        self.cam.stop_recording(save_to_filename=top_path, fps=fps)
        self.head_cam.stop_recording(save_to_filename=head_path, fps=fps)
        self._recording = False
        print(f"✓ Saved top-down video to {top_path}")
        print(f"✓ Saved head-camera video to {head_path}")
        return top_path, head_path

    def close(self):
        """Flush any in-progress recording."""
        if self._recording:
            self.save_video()

    # convenience demo ------------------------------------------------
    def demo(self, video_path=None):
        """A short smoke-test: drive a small path and do one IK reach with the left arm.

        Used by the CLI; the example notebook is the full usage guide. If the caller
        started recording (``start_recording()``) beforehand, the run is written to
        ``video_path`` (or a timestamped default) at the end.
        """
        self.set_base_velocity(LINEAR_SPEED_MPS, 0.0, 0.0, steps=200)     # forward
        self.set_base_velocity(0.0, 0.0, ANGULAR_SPEED_RADPS, steps=120)  # rotate in place
        self.set_base_velocity(0.0, LINEAR_SPEED_MPS, 0.0, steps=120)     # strafe left
        self.stop_base(steps=40)
        # one IK reach in front of the left arm, then back to the tuck pose
        pos, _ = self.get_base_pose()
        self.move_ee("left", (pos[0] + 0.45, pos[1] + 0.3, 0.95), n_waypoints=40)
        self.set_arm("left", [ARM_HOLD[n] for n in self.arm_joint_names["left"]])
        self.step(60)
        if self._recording:
            self.save_video(video_path)
        return self


########################## command-line entry point ##########################
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the Genesis robot scene and run a short demo.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--headless", dest="headless", action="store_true",
                   help="Force headless (no viewer). Default: auto-detect from DISPLAY.")
    g.add_argument("--no-headless", dest="headless", action="store_false",
                   help="Force the interactive viewer on.")
    p.set_defaults(headless=None)
    p.add_argument("--save-video", action="store_true", help="Record the demo to an mp4.")
    p.add_argument("--video-path", default=None, help="Output mp4 path (default: timestamped).")
    p.add_argument("--backend", choices=["auto", "cpu", "cuda", "amd", "metal"], default="auto",
                   help="Genesis backend (default: auto-select).")
    return p.parse_args(argv)


def _resolve_backend(name):
    return _backend_from_name(name)


def main(argv=None):
    args = _parse_args(argv)
    sim = RobotScene(
        headless=args.headless,
        backend=_resolve_backend(args.backend),
    )
    if args.save_video:
        sim.start_recording()
    print("=" * 80)
    print("✓ Scene built (11 tables + 9 letters + 3 cutlery + 1 mobile dual-arm robot)")
    print("  Running the smoke-test demo; see notebooks/robot_scene_demo.ipynb for full usage.")
    print("=" * 80)
    sim.demo(video_path=args.video_path)
    print("✓ Done!")


if __name__ == "__main__":
    main()
