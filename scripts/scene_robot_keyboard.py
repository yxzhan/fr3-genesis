#!/usr/bin/env python3
"""
完整场景 + 键盘控制(Genesis 版本):10个桌子 + 9个字母 + 3个餐具 + 1个可控机器人

这是 scripts/scenes/scene_robot_keyboard.py(Isaac Lab) 的 Genesis 移植版,
本质上是把两个已经移植好的 Genesis 例子合在一起:
  * 场景布置(桌子/字母/餐具)来自 scene_robot_tables.py
  * 底盘运动学 + 键盘控制来自 keyboard_control.py

和静态展示场景(scene_robot_tables.py)最大的区别:
  机器人底座这里必须是自由的(fixed=False),才能被驱动轮推着在场景里跑。
  机器人初始放在顶部中间 (0, 4.5),用 WASD/QE 开着它在桌子之间穿行。

键位(WASD 与 Genesis viewer 快捷键冲突,故平移用方向键、旋转用 Q/E):
  ↑/↓ 前后, ←/→ 左右平移, Q/E 原地左转/右转, 可组合(如 ↑+←),ESC 退出。

依赖: pip install pynput  (全局键盘监听,和原脚本一致)

坐标 / 四元数约定: Genesis 与 Isaac Lab 都是 Z-up、米、四元数 w-x-y-z,位姿原样照搬。
"""

import math
import os

import numpy as np
import genesis as gs
from pynput import keyboard

########################## 底盘运动学常量 ##########################
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

# 每个驱动模块: (转向关节, 驱动轮关节, 机体系 x, 机体系 y)。ROS 约定: +x 前, +y 左。
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
        # 顶部斜俯视,能看到整片桌子网格和机器人
        camera_pos=(0.0, -7.0, 14.0),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
        max_FPS=60,
    ),
    # 小步长提高稳定性(和 keyboard_control.py 一致)
    sim_options=gs.options.SimOptions(dt=0.005, gravity=(0.0, 0.0, -9.81)),
    show_viewer=True,
)

########################## 地面 ##########################
# 大幅提高摩擦,给驱动轮足够抓地力(原脚本 static=2.0 / dynamic=1.5)
scene.add_entity(
    gs.morphs.Plane(),
    material=gs.materials.Rigid(friction=2.0),
)

########################## 桌子(11 张) ##########################
# 桌子已从 table_edit.usd 转成 MJCF。转换时已把 90°(绕 X)的朝向烘焙进 body 的
# quat,所以这里只传 pos。body 没有 joint => 默认焊死在世界系(等价于 fixed)。
table_mjcf = os.path.join(SCRIPT_DIR, "mjcf", "table_edit", "table_edit.xml")

# 左列 5 张 (X=-2.0) + 右列 5 张 (X=2.0) + 底部中间 1 张。
# 顶部中间 (0, 4.5) 留给机器人。
table_positions = [
    (-2.0, 3.0, 0.0), (-2.0, 1.5, 0.0), (-2.0, 0.0, 0.0), (-2.0, -1.5, 0.0), (-2.0, -3.0, 0.0),
    (2.0, 3.0, 0.0), (2.0, 1.5, 0.0), (2.0, 0.0, 0.0), (2.0, -1.5, 0.0), (2.0, -3.0, 0.0),
    (0.0, -4.5, 0.0),  # 底部中间
]
for pos in table_positions:
    scene.add_entity(
        gs.morphs.MJCF(file=table_mjcf, pos=pos),
    )

########################## 字母(9 个,黑色) ##########################
# 字母资产已从 *_edit.usd 转成 MJCF(mjcf/<L>_edit/<L>_edit.xml)。MJCF body 里已烘焙
# 了各自的朝向,但字母还需要再绕 X 轴转 90° 才能立起来(和原 USD 场景叠加的 quat 一致),
# 这个 morph.quat 会叠加在 body 已有朝向之上。body 无 joint => 焊死(等价 fixed)。
# 颜色用 surface 强制成黑色。
LETTER_BLACK = gs.surfaces.Default(color=(0.0, 0.0, 0.0))
LETTER_QUAT = (0.7071, 0.7071, 0.0, 0.0)  # 绕 X 轴 90°
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

########################## 餐具(3 个) ##########################
# 餐具资产已转成 MJCF(mjcf/<item>/<item>.xml),朝向同样烘焙进 body quat,只传 pos。
# 原 Isaac Lab 里的 offset 把餐具放到了桌面外/桌面下(z 基准用的是 0.0 而非桌面 0.7),
# 所以会掉到地上。这里把基准抬到桌面高度 0.7,并把 x/y offset 收到桌子中心附近。
ikea_table_pos = (-2.0, 3.0, 0.7)  # 餐具桌的桌面中心
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

########################## 机器人(可控,顶部中间) ##########################
# 底座必须是自由的(fixed=False)才能被轮子推着走。
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


# ---- 关节 / dof 句柄 ----
def dof(name):
    return robot.get_joint(name).dofs_idx_local[0]


# 机械臂(双臂 7+7)与夹爪(双手 2+2)
arm_joint_names = [f"{side}_fr3v2_joint{i}" for side in ("left", "right") for i in range(1, 8)]
finger_joint_names = [f"{side}_fr3v2_finger_joint{j}" for side in ("left", "right") for j in (1, 2)]
arm_dofs = [dof(n) for n in arm_joint_names]
finger_dofs = [dof(n) for n in finger_joint_names]

# 主动底盘关节
steering_dofs = [dof(m[0]) for m in DRIVE_MODULES]
drive_dofs = [dof(m[1]) for m in DRIVE_MODULES]

# 被动关节(万向轮转向 + 万向轮滚动 + 后摇臂)—— 不伺服,任其自由
passive_joint_names = [
    "caster_front_left_steering_joint", "caster_front_left_joint",
    "caster_rear_right_steering_joint", "caster_rear_right_joint",
    "rocker_arm_joint",
]
passive_dofs = [dof(n) for n in passive_joint_names]

# ---- 控制增益 ----
# 机械臂:超高刚度/阻尼,把手臂牢牢锁在初始姿态
robot.set_dofs_kp(np.array([5000.0] * len(arm_dofs)), arm_dofs)
robot.set_dofs_kv(np.array([500.0] * len(arm_dofs)), arm_dofs)
robot.set_dofs_force_range(np.array([-200.0] * len(arm_dofs)), np.array([200.0] * len(arm_dofs)), arm_dofs)
# 夹爪:位置控制
robot.set_dofs_kp(np.array([200.0] * len(finger_dofs)), finger_dofs)
robot.set_dofs_kv(np.array([20.0] * len(finger_dofs)), finger_dofs)
robot.set_dofs_force_range(np.array([-50.0] * len(finger_dofs)), np.array([50.0] * len(finger_dofs)), finger_dofs)
# 转向关节:位置控制,高刚度保持转角
robot.set_dofs_kp(np.array([500.0] * len(steering_dofs)), steering_dofs)
robot.set_dofs_kv(np.array([50.0] * len(steering_dofs)), steering_dofs)
robot.set_dofs_force_range(np.array([-200.0] * len(steering_dofs)), np.array([200.0] * len(steering_dofs)), steering_dofs)
# 驱动轮:速度控制(kp=0,靠 kv 跟踪目标轮速)
robot.set_dofs_kp(np.array([0.0] * len(drive_dofs)), drive_dofs)
robot.set_dofs_kv(np.array([50.0] * len(drive_dofs)), drive_dofs)
robot.set_dofs_force_range(np.array([-500.0] * len(drive_dofs)), np.array([500.0] * len(drive_dofs)), drive_dofs)
# 被动关节:零增益,自由转动
robot.set_dofs_kp(np.array([0.0] * len(passive_dofs)), passive_dofs)
robot.set_dofs_kv(np.array([0.0] * len(passive_dofs)), passive_dofs)

# ---- 初始关节姿态(双臂收起,和原脚本 initial_joint_pos 一致) ----
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

# 让机器人在地面上稳定下来(底座自由,会先落稳)
for _ in range(200):
    robot.control_dofs_position(arm_hold_targets, arm_dofs)
    robot.control_dofs_position(finger_open, finger_dofs)
    scene.step()


########################## 底盘运动学(numpy 版) ##########################
def _wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def _steering_alignment_scale(error):
    """转向还没到位时,把轮速渐隐下来,避免侧滑。"""
    scale = (STEERING_ZERO_SPEED_ERROR_RAD - error) / (
        STEERING_ZERO_SPEED_ERROR_RAD - STEERING_FULL_SPEED_ERROR_RAD
    )
    return max(0.0, min(1.0, scale))


def get_keyboard_twist(pressed):
    """按键 -> 机体系 (vx, vy, wz)。

    注意:WASD 与 Genesis viewer 的快捷键冲突,所以平移改用方向键,旋转用 Q/E
    (q/e 不和 viewer 冲突)。
        ↑/↓  前进/后退      ←/→  左移/右移(平移)      Q/E  原地左转/右转
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
    """机体 twist -> (转向角目标, 轮速目标),含 180° 翻转优化与限速。"""
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
            steer_targets[i] = cur  # 停时保持当前转角,别乱回正
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
    """纯平移时锁航向,手动旋转/静止时重置保持航向。"""
    cur = get_root_yaw()
    if manual_rotation or math.hypot(vx, vy) < STOP_EPS:
        return wz, cur
    err = _wrap_to_pi(desired_yaw - cur)
    comp = HEADING_HOLD_KP * err - HEADING_HOLD_KD * get_root_yaw_rate()
    comp = max(-MAX_HEADING_COMP_RADPS, min(MAX_HEADING_COMP_RADPS, comp))
    return wz + comp, desired_yaw


########################## 键盘监听(pynput,全局) ##########################
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
print("✓ 开始仿真!  (Genesis 完整场景 + 键盘控制)")
print("  - 11 张桌子 + 9 个字母 + 3 个餐具 + 1 个可控机器人(顶部中间)")
print("=" * 80)
print("控制: ↑/↓ 前后 | ←/→ 左右平移 | Q/E 原地左转/右转 | 可组合 | ESC 退出")
print("=" * 80)

########################## 控制循环 ##########################
heading_hold_yaw = get_root_yaw()
count = 0
try:
    while True:
        vx, vy, wz_cmd = get_keyboard_twist(_pressed)
        wz, heading_hold_yaw = compensate_yaw_rate(
            vx, vy, wz_cmd, heading_hold_yaw, manual_rotation=abs(wz_cmd) > 1.0e-4
        )

        # 机械臂 + 夹爪:锁在初始姿态
        robot.control_dofs_position(arm_hold_targets, arm_dofs)
        robot.control_dofs_position(finger_open, finger_dofs)

        # 底盘:转向用位置控制、驱动轮用速度控制
        cur_steer = robot.get_dofs_position(steering_dofs).cpu().numpy()
        steer_targets, drive_targets = compute_drive_targets(cur_steer, vx, vy, wz)
        robot.control_dofs_position(steer_targets, steering_dofs)
        robot.control_dofs_velocity(drive_targets, drive_dofs)

        scene.step()

        count += 1
        if count % 100 == 0 and (vx or vy or wz):
            pos = robot.get_pos().cpu().numpy()
            print(
                f"步数 {count} | vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} | "
                f"位置 [{pos[0]:.2f}, {pos[1]:.2f}] 朝向 {math.degrees(get_root_yaw()):.1f}°"
            )
except KeyboardInterrupt:
    print("\n✓ 用户停止")
finally:
    if _listener is not None:
        _listener.stop()
    print("✓ 完成!")
