# Parallel / batched `RobotScene` — design

**Date:** 2026-06-04
**Status:** approved (pending spec review)
**Files touched:**
- `scripts/scene_robot_tables.py` — all batching logic, incl. the CLI
  (`_parse_args` / `main` / `demo`), lives in this one module.
- `notebooks/parallel_demo.ipynb` — **new** demo notebook (see Deliverables).
- `notebooks/robot_scene_demo.ipynb` — **unchanged** (the `n_envs=1` scalar path
  preserves its current behavior).

## Goal

Let `RobotScene` run `N` copies of the full scene (furniture + mobile dual-arm
robot) in parallel via Genesis `scene.build(n_envs=N)`, with **all** control —
including the mobile-base heading-hold / steering loop — vectorized over the
batch dimension. Mirrors the batching pattern in
`PhySim04_parallel_simulation.ipynb` (`scene.build(n_envs=N, env_spacing=...)`,
control arrays carry a leading `[N, ...]` dim, one `scene.step()` advances all
envs).

## Decisions (locked)

1. **Scope:** full `RobotScene × N`; vectorize every control path incl. the base.
2. **Back-compat:** `n_envs=1` (default) reproduces today's exact scalar API and
   builds Genesis **non-batched** (`scene.build()` with no `n_envs`). The
   committed `robot_scene_demo.ipynb` keeps working unchanged. The batched
   `[N, ...]` API only activates when `n_envs > 1`.
3. **Rendering:** top-down camera frames the whole env grid as one image; the
   attached head camera follows **env 0**. No per-env image extraction.
4. **Approach:** unified batched core + scalar boundary (Approach A below).

## Approach A — unified batched core, scalar boundary

Represent all base-controller state internally as batched `[N, ...]` with `N ≥ 1`.
Write the math helpers once, vectorized over a leading axis. A thin adapter
squeezes/unsqueezes the batch dim at two boundaries:

- **Genesis I/O boundary:** in non-batched mode (`n_envs==1`) read with
  `np.atleast_2d` and squeeze on write; in batched mode pass through. So the
  core always sees `[N, ...]`.
- **Public boundary:** getters return `[N, ...]` when batched, else squeeze to
  today's scalar / `[d]`; setters accept a scalar/`[d]` (broadcast to all envs)
  **or** a full `[N, ...]`.

When `n_envs==1` Genesis is built non-batched exactly as today → zero physics or
array-shape regression to the committed demo. (Rejected alternative B — always
build batched even for N=1 — changes the default Genesis mode and makes
`self.cube.get_pos()` return `[1,3]`, breaking the notebook.)

## Components

### State (`__init__` / `_setup_handles`)
- `self.n_envs: int` (≥1), `self.batched: bool = n_envs > 1`.
- `self._twist` → `np.ndarray [N, 3]` (was a 3-tuple).
- `self._heading_hold_yaw` → `np.ndarray [N]` (was a float).
- `self._arm_target[side]` → `[N, 7]`, `self._finger_target[side]` → `[N, 2]`,
  `self._spine_target` → `[N]`. A scalar/`[d]` set broadcasts across envs.

### Vectorized math core (the real work)
- `compute_drive_targets(cur_steer[B,4], vx[B], vy[B], wz[B]) -> (steer[B,4], drive[B,4])`:
  rewrite the per-drive-module scalar loop as numpy broadcasting over the
  `4 modules × B envs` grid. Preserve the existing semantics:
  - per-module world velocity `wvx = vx - wz*y`, `wvy = vy + wz*x`;
  - global speed-limit scaling from `MAX_WHEEL_SPEED_RADPS * WHEEL_RADIUS_M`
    (per-env max over modules);
  - 180°-flip optimization (`direct` vs `flipped`, choose smaller `|delta|`);
  - hold current steer angle where module speed `< STOP_EPS`;
  - `_steering_alignment_scale` fade on wheel speed.
- `_wrap_to_pi`, `_steering_alignment_scale` → numpy, array-friendly
  (`np.arctan2`, `np.clip`), still valid for scalar/`[1]` inputs.
- yaw extraction from quaternion → vectorized over `[N,4]`.

### Per-step control
- `_apply_base_control`: read `cur_yaw[N]`, `yaw_rate[N]`, `cur_steer[N,4]`;
  apply heading-hold per env (`manual_rotation` / near-zero-twist branch becomes
  an `np.where` mask); issue batched `control_dofs_position(steer[N,4])` /
  `control_dofs_velocity(drive[N,4])`.
- `_apply_joint_targets`: issue batched spine/arm/finger position targets.
- `_apply_initial_pose`: build `q_init[N, n_dofs]`, set per-dof columns, settle.

### Genesis I/O adapter (private helpers)
Thin wrappers used by the core so it always sees `[N, ...]`:
`get_dofs_position`, `get_pos/get_quat/get_ang`, `set_dofs_position/set_pos/set_quat`,
`inverse_kinematics`. Non-batched: `atleast_2d` on read, `[0]`/squeeze on write.

### Public boundary
- **Getters** (`get_pos`, `get_yaw`, `get_yaw_rate`, `get_arm`, `get_ee_pos`,
  `get_spine`, `get_base_pose`): `[N, ...]` when batched, else squeeze to scalar/`[d]`.
- **Setters** (`set_arm`, `set_gripper`, `set_spine`, `set_base_velocity`,
  `stop_base`, `teleport_base`, `reset_pose`): accept scalar/`[d]` (broadcast) or `[N, ...]`.
- **IK** (`ik`, `move_ee`): accept `pos[N,3]`, `quat[N,4]`; `move_ee` interpolates
  per-env start→target straight lines; `ik` indexes `[:, arm_qs]` (batched) or
  `[arm_qs]` (scalar).

### Build / scene
- `__init__(..., n_envs=1, env_spacing=(2.0, 2.0))`.
- `_build_scene` unchanged — Genesis auto-replicates all entities per env with
  the spacing offset.
- `scene.build(n_envs=N, env_spacing=env_spacing)` when `N>1`, else `scene.build()`.

### Rendering
- Top-down `cam` pulled back/up so the whole `env_spacing × n_envs` grid fits
  (computed from spacing + env count, or caller-overridable via existing
  `camera_res` plus new framing); `render()` returns one grid image.
- `head_cam.attach` binds to env 0's head link; `render_head()` = env 0 view.
- Recording (`start_recording`/`save_video`) unchanged.

### CLI
- Add `--n-envs N` (default 1).
- `demo()`: when batched, issue per-env-varied commands (e.g. random per-env
  `angular_speed`/twist, à la PhySim04) to visibly exercise parallelism;
  unchanged single-env path when `N==1`.

## Deliverables

1. Batched `scripts/scene_robot_tables.py` (above).
2. **New** `notebooks/parallel_demo.ipynb`, modeled on PhySim04:
   `RobotScene(n_envs=9, ...)`, assign each env a different base/arm command
   (e.g. random `angular_speed`), step, record the grid video, embed it inline.
   Imports via the package (`from fr3_genesis import RobotScene`), like the
   existing notebook.
3. `notebooks/robot_scene_demo.ipynb` stays as-is (regression target).

## Data flow (batched step)

```
user sets per-env targets  ->  self._twist[N,3], self._arm_target[N,7], ...
step() loop, each tick:
  _apply_base_control:  read state[N] -> compute_drive_targets -> control_dofs_*[N,*]
  _apply_joint_targets: control_dofs_position(spine/arm/finger)[N,*]
  scene.step()                      # advances all N envs
  (record) cam.render() + head_cam(env0).render()
```

## Error handling

- Setter input shape: accept scalar, `[d]`, or `[N, d]`; raise a clear
  `ValueError` on any other shape (e.g. `[M, d]`, `M != N`).
- `n_envs` must be a positive int; `n_envs == 1` selects the scalar path.

## Testing

- **Back-compat:** `n_envs=1` — `get_arm("left").shape == (7,)`, `get_yaw()`
  returns a float, `get_base_pose()[0].shape == (3,)`; a short drive+IK matches
  pre-change behavior (smoke test, no exception, cube reachable).
- **Batched shapes:** `n_envs=4` — getters return leading dim 4; `set_arm` with
  `[4,7]` and with `[7]` (broadcast) both run; `move_ee` with `[4,3]` runs.
- **Per-env divergence:** drive each env with a different `wz`; assert the four
  `get_yaw()` values diverge after stepping (parallelism actually independent).
- **compute_drive_targets equivalence:** for `N` random twists, the vectorized
  output equals the old scalar function applied env-by-env (within tolerance).

## Out of scope (YAGNI)

Per-env independent camera framing, per-env reset/done bookkeeping, a gym/RL env
wrapper, changing the furniture layout per env.
