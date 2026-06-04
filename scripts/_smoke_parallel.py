"""Manual integration smoke test for batched RobotScene. Run on a GPU machine.

    uv run python scripts/_smoke_parallel.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_robot_tables import RobotScene


def check_single_env():
    sim = RobotScene(headless=True, n_envs=1)
    arm = sim.get_arm("left")
    yaw = sim.get_yaw()
    pos, yaw2 = sim.get_base_pose()
    assert arm.shape == (7,), f"n_envs=1 get_arm shape {arm.shape}"
    assert np.isscalar(yaw) or np.ndim(yaw) == 0, f"n_envs=1 get_yaw not scalar: {yaw!r}"
    assert pos.shape == (3,), f"n_envs=1 get_pos shape {pos.shape}"
    sim.set_base_velocity(0.5, 0.0, 0.0, steps=50)  # scalar twist must still work
    print("[n_envs=1] OK: scalar API preserved")


def check_batched_env():
    n = 4
    sim = RobotScene(headless=True, n_envs=n)
    arm = sim.get_arm("right")
    assert arm.shape == (n, 7), f"batched get_arm shape {arm.shape}"
    assert sim.get_pos().shape == (n, 3), "batched get_pos shape"
    assert sim.get_yaw().shape == (n,), "batched get_yaw shape"

    # broadcast setter: one [7] target to all envs
    sim.set_arm("right", arm[0])
    # per-env setter: full [n,7]
    sim.set_arm("right", arm)

    # Per-env divergence: give each env a different yaw rate, step, expect spread.
    yaw0 = sim.get_yaw().copy()
    sim.set_base_velocity(0.0, 0.0, np.array([-1.2, -0.4, 0.4, 1.2]), steps=120)
    yaw1 = sim.get_yaw()
    spread = float(np.ptp(yaw1 - yaw0))
    assert spread > 0.2, f"envs did not diverge under different wz (spread={spread:.3f})"
    print(f"[n_envs={n}] OK: batched shapes + per-env divergence (yaw spread={spread:.3f})")


if __name__ == "__main__":
    check_single_env()
    check_batched_env()
    print("SMOKE OK")
