# Parallel / batched RobotScene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `RobotScene` run `N` copies of the full scene in parallel via `scene.build(n_envs=N)`, with all control — including the mobile-base steering/heading-hold loop — vectorized over the batch dim; `n_envs=1` (default) keeps today's exact scalar API.

**Architecture:** Unified batched core + scalar boundary. The drive/yaw math is rewritten as numpy functions that operate over a leading batch axis `[B, …]`. `RobotScene` holds base-controller state as `[N, …]` (N≥1). A thin adapter squeezes/unsqueezes the batch dim at the Genesis I/O boundary; public getters/setters squeeze to scalar when `n_envs==1`. When `n_envs==1` Genesis is built non-batched (exactly as today) → zero regression.

**Tech Stack:** Python, numpy, Genesis (`genesis-world==0.4.6`), uv, pytest (new dev dep). Spec: `docs/superpowers/specs/2026-06-04-parallel-batched-robotscene-design.md`.

**Conventions used below:** `M = len(DRIVE_MODULES)` (drive modules, currently 2). All edits are in `scripts/scene_robot_tables.py` unless stated. Pure-math unit tests run on CPU (no GPU); Genesis scene tests are a manual smoke step (Task 8) because they need a built scene.

---

## Task 1: Add pytest dev dependency and a tests directory

**Files:**
- Modify: `pyproject.toml` (the `[dependency-groups]` block)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest to the dev group**

In `pyproject.toml`, change the existing dev group:

```toml
[dependency-groups]
dev = ["ipykernel", "pytest"]
```

- [ ] **Step 2: Make `scripts/` importable in tests**

Create `tests/conftest.py`:

```python
import os
import sys

# scene_robot_tables.py lives in scripts/; make it importable as a top-level module.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
```

Create empty `tests/__init__.py`:

```python
```

- [ ] **Step 3: Sync and verify pytest is available**

Run: `uv sync --extra cu126 && uv run pytest --version`
Expected: prints `pytest 8.x` (or similar), no error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py tests/conftest.py
git commit -m "test: add pytest dev dependency and tests scaffold"
```

---

## Task 2: Vectorize `_wrap_to_pi` and `_steering_alignment_scale`

These are pure helpers. Rewrite them with numpy so they accept scalars **or** arrays (broadcasting), while returning the same values for scalar input.

**Files:**
- Modify: `scripts/scene_robot_tables.py:229-238`
- Create: `tests/test_drive_math.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_drive_math.py`:

```python
import math

import numpy as np

import scene_robot_tables as s


def test_wrap_to_pi_scalar_matches_math():
    for a in [0.0, 3.0, -3.0, 7.0, -7.0, math.pi, -math.pi]:
        expected = math.atan2(math.sin(a), math.cos(a))
        assert np.isclose(s._wrap_to_pi(a), expected)


def test_wrap_to_pi_array_broadcasts():
    a = np.array([0.0, 7.0, -7.0, math.pi + 0.1])
    out = s._wrap_to_pi(a)
    assert out.shape == a.shape
    assert np.all(out >= -math.pi - 1e-9) and np.all(out <= math.pi + 1e-9)


def test_steering_alignment_scale_clamps_and_fades():
    # error >= ZERO threshold -> 0; error <= FULL threshold -> 1; linear between.
    assert np.isclose(s._steering_alignment_scale(s.STEERING_ZERO_SPEED_ERROR_RAD), 0.0)
    assert np.isclose(s._steering_alignment_scale(s.STEERING_FULL_SPEED_ERROR_RAD), 1.0)
    assert np.isclose(s._steering_alignment_scale(0.0), 1.0)  # clamped to 1
    assert np.isclose(s._steering_alignment_scale(math.pi), 0.0)  # clamped to 0


def test_steering_alignment_scale_array():
    err = np.array([0.0, s.STEERING_FULL_SPEED_ERROR_RAD, s.STEERING_ZERO_SPEED_ERROR_RAD, math.pi])
    out = s._steering_alignment_scale(err)
    assert out.shape == err.shape
    assert np.allclose(out, [1.0, 1.0, 0.0, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_drive_math.py -v`
Expected: `test_wrap_to_pi_array_broadcasts` and `test_steering_alignment_scale_array` FAIL (the scalar `math.*`/`max`/`min` versions raise or return wrong shape on arrays).

- [ ] **Step 3: Rewrite the helpers with numpy**

Replace `scripts/scene_robot_tables.py:229-238` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_drive_math.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_robot_tables.py tests/test_drive_math.py
git commit -m "refactor: make _wrap_to_pi / _steering_alignment_scale array-friendly"
```

---

## Task 3: Vectorize `compute_drive_targets` over a batch axis

Rewrite the per-module scalar loop as numpy broadcasting over `[B, M]`. The function now takes batched inputs: `cur_steer[B, M]`, `vx[B]`, `vy[B]`, `wz[B]`, and returns `steer[B, M]`, `drive[B, M]`. Semantics preserved exactly (per-env speed-limit scaling, 180°-flip optimization, hold-steer-when-stopped, alignment fade).

**Files:**
- Modify: `scripts/scene_robot_tables.py:241-275`
- Modify: `tests/test_drive_math.py`

- [ ] **Step 1: Write the failing test (with a frozen reference of the old scalar logic)**

Append to `tests/test_drive_math.py`:

```python
import math as _math


def _ref_scalar_drive_targets(cur_steer_angles, vx, vy, wz):
    """Frozen copy of the ORIGINAL scalar compute_drive_targets, used as the
    equivalence oracle for the vectorized version."""
    steer_targets = np.zeros(len(s.DRIVE_MODULES))
    drive_targets = np.zeros(len(s.DRIVE_MODULES))
    vectors = []
    max_speed = 0.0
    for (_s, _d, x, y) in s.DRIVE_MODULES:
        wvx = vx - wz * y
        wvy = vy + wz * x
        sp = _math.hypot(wvx, wvy)
        vectors.append((wvx, wvy, sp))
        max_speed = max(max_speed, sp)
    allowed = s.MAX_WHEEL_SPEED_RADPS * s.WHEEL_RADIUS_M
    scale = allowed / max_speed if max_speed > allowed else 1.0
    for i, (wvx, wvy, sp) in enumerate(vectors):
        wvx *= scale
        wvy *= scale
        sp *= scale
        cur = cur_steer_angles[i]
        if sp < s.STOP_EPS:
            steer_targets[i] = cur
            continue
        raw = _math.atan2(wvy, wvx)
        direct = _math.atan2(_math.sin(raw - cur), _math.cos(raw - cur))
        flipped = _math.atan2(_math.sin(raw + _math.pi - cur), _math.cos(raw + _math.pi - cur))
        use_flipped = abs(flipped) < abs(direct)
        delta = flipped if use_flipped else direct
        steer_targets[i] = cur + delta
        align = max(0.0, min(1.0, (s.STEERING_ZERO_SPEED_ERROR_RAD - abs(delta)) /
                              (s.STEERING_ZERO_SPEED_ERROR_RAD - s.STEERING_FULL_SPEED_ERROR_RAD)))
        wheel_speed = (sp / s.WHEEL_RADIUS_M) * align
        drive_targets[i] = -wheel_speed if use_flipped else wheel_speed
    return steer_targets, drive_targets


def test_compute_drive_targets_batched_matches_scalar_reference():
    rng = np.random.default_rng(0)
    M = len(s.DRIVE_MODULES)
    twists = [
        (0.0, 0.0, 0.0),       # stopped
        (0.5, 0.0, 0.0),       # forward
        (0.0, 0.5, 0.0),       # strafe
        (0.0, 0.0, 1.2),       # rotate
        (2.0, -1.0, 0.7),      # fast mixed (triggers speed limit)
    ]
    cur = rng.uniform(-math.pi, math.pi, size=(len(twists), M))
    vx = np.array([t[0] for t in twists])
    vy = np.array([t[1] for t in twists])
    wz = np.array([t[2] for t in twists])

    steer, drive = s.compute_drive_targets(cur, vx, vy, wz)
    assert steer.shape == (len(twists), M)
    assert drive.shape == (len(twists), M)

    for i, (tvx, tvy, twz) in enumerate(twists):
        ref_steer, ref_drive = _ref_scalar_drive_targets(cur[i], tvx, tvy, twz)
        assert np.allclose(steer[i], ref_steer, atol=1e-9), f"steer env {i}"
        assert np.allclose(drive[i], ref_drive, atol=1e-9), f"drive env {i}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_drive_math.py::test_compute_drive_targets_batched_matches_scalar_reference -v`
Expected: FAIL (old function does `np.zeros(M)` and scalar `math.hypot`, so batched input raises or mis-shapes).

- [ ] **Step 3: Rewrite `compute_drive_targets` vectorized**

Replace `scripts/scene_robot_tables.py:241-275` with:

```python
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
```

- [ ] **Step 4: Run the full math test file**

Run: `uv run pytest tests/test_drive_math.py -v`
Expected: all tests PASS (including the equivalence test).

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_robot_tables.py tests/test_drive_math.py
git commit -m "refactor: vectorize compute_drive_targets over the env batch axis"
```

---

## Task 4: Add a vectorized `quat_to_yaw` helper

Extract yaw-from-quaternion into a module function that works on `[B, 4]` (and a single `[4]`), so both the scalar `get_yaw` and the batched base controller share one implementation.

**Files:**
- Modify: `scripts/scene_robot_tables.py` (add near the other math helpers, after `_steering_alignment_scale`)
- Modify: `tests/test_drive_math.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_drive_math.py`:

```python
def test_quat_to_yaw_scalar_and_batched():
    # Identity quat -> yaw 0. 90 deg about z -> yaw pi/2. Quats are (w, x, y, z).
    import math
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * (math.pi / 2)
    q_90 = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])

    assert np.isclose(s.quat_to_yaw(q_id), 0.0)
    assert np.isclose(s.quat_to_yaw(q_90), math.pi / 2)

    batched = np.stack([q_id, q_90])
    out = s.quat_to_yaw(batched)
    assert out.shape == (2,)
    assert np.allclose(out, [0.0, math.pi / 2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_drive_math.py::test_quat_to_yaw_scalar_and_batched -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'quat_to_yaw'`.

- [ ] **Step 3: Add the helper**

Add after `_steering_alignment_scale` in `scripts/scene_robot_tables.py`:

```python
def quat_to_yaw(quat):
    """Yaw (rad) about world z from a (w, x, y, z) quaternion. Accepts [4] or [B, 4]."""
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_drive_math.py::test_quat_to_yaw_scalar_and_batched -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_robot_tables.py tests/test_drive_math.py
git commit -m "feat: add vectorized quat_to_yaw helper"
```

---

## Task 5: Add `n_envs` / `env_spacing` to `__init__` and the build call

Introduce the batch dimension at construction. Genesis builds non-batched for `n_envs==1` (today's behavior) and batched for `n_envs>1`.

**Files:**
- Modify: `scripts/scene_robot_tables.py:306-316` (`__init__` signature + state)
- Modify: `scripts/scene_robot_tables.py:355` (the `self.scene.build()` call)

- [ ] **Step 1: Extend the `__init__` signature and store batch state**

Replace the signature and the first lines of `__init__` (`scripts/scene_robot_tables.py:306-316`):

```python
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
```

- [ ] **Step 2: Build batched when requested**

Replace `self.scene.build()` at `scripts/scene_robot_tables.py:355` with:

```python
        if self.batched:
            self.scene.build(n_envs=self.n_envs, env_spacing=self.env_spacing)
        else:
            self.scene.build()
```

- [ ] **Step 3: Add the docstring lines for the new params**

In the `__init__` docstring (the `Parameters` block around `scripts/scene_robot_tables.py:286-304`), add before the closing `"""`:

```
    n_envs : int
        Number of parallel environments. 1 (default) builds a single non-batched
        scene with the classic scalar API. >1 builds ``n_envs`` copies; getters
        then return a leading ``[n_envs, ...]`` dim and setters accept either a
        scalar/per-dof value (broadcast to all envs) or a full ``[n_envs, ...]`` array.
    env_spacing : (float, float)
        Grid spacing between parallel envs (only used when n_envs > 1).
```

- [ ] **Step 4: Verify the module still imports**

Run: `uv run python -c "import sys; sys.path.insert(0,'scripts'); import scene_robot_tables; print('ok')"`
Expected: prints `ok` (no GPU needed; `gs.init` is not called on import).

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_robot_tables.py
git commit -m "feat: add n_envs / env_spacing to RobotScene.__init__"
```

---

## Task 6: Batch the controller state, Genesis I/O adapter, and per-step control

Make the base-controller state batched `[N, …]`, route all Genesis reads through an adapter that guarantees a `[N, …]` view, and issue batched control. The arm/gripper/spine targets become `[N, …]` too.

**Files:**
- Modify: `scripts/scene_robot_tables.py` — `_setup_handles` target init (`:480-482`), `_apply_initial_pose` (`:516-527`), `_apply_joint_targets` (`:530-535`), `_apply_base_control` (`:537-554`), and the base state set in `__init__` (`:366-368`). Add private I/O helpers.

- [ ] **Step 1: Add Genesis I/O adapter helpers**

Add these methods to the `RobotScene` class (e.g. right after `_setup_gains`):

```python
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
```

- [ ] **Step 2: Batch the target state in `_setup_handles`**

Replace `scripts/scene_robot_tables.py:480-482` with:

```python
        # Current control targets (updated by set_arm / set_gripper, re-applied each
        # step). Stored batched as [N, width] so each env can hold a different target.
        self._arm_target = {
            s: np.tile(np.array([ARM_HOLD[n] for n in self.arm_joint_names[s]]), (self.n_envs, 1))
            for s in ("left", "right")
        }
        self._finger_target = {s: np.full((self.n_envs, 2), GRIPPER_OPEN) for s in ("left", "right")}
        self._spine_target = np.full(self.n_envs, SPINE_HOLD)
```

Note: this replaces the scalar `self._spine_target = SPINE_HOLD` set in `_setup_handles` (`:469`). Delete that line (`self._spine_target = SPINE_HOLD`) so the batched version above is the only initializer.

- [ ] **Step 3: Batch `_apply_joint_targets`**

Replace `scripts/scene_robot_tables.py:530-535` with:

```python
    def _apply_joint_targets(self):
        """Re-issue the stored arm + gripper + spine position targets (every sim step)."""
        self.robot.control_dofs_position(self._emit(self._spine_target[:, None]), [self.spine_dof])
        for s in ("left", "right"):
            self.robot.control_dofs_position(self._emit(self._arm_target[s]), self.arm_dofs[s])
            self.robot.control_dofs_position(self._emit(self._finger_target[s]), self.finger_dofs[s])
```

- [ ] **Step 4: Batch the base-controller state in `__init__`**

Replace `scripts/scene_robot_tables.py:366-368` with:

```python
        # Base heading-hold state (batched [N, ...]), reset after settling.
        self._twist = np.zeros((self.n_envs, 3))
        self._heading_hold_yaw = self._yaw_all()
```

- [ ] **Step 5: Batch `_apply_base_control` and add `_yaw_all` / `_yaw_rate_all`**

Add these helpers to the class:

```python
    def _yaw_all(self):
        """Yaw of every env as [N] (uses the batched quaternion read)."""
        return quat_to_yaw(self._read(self.robot.get_quat()))

    def _yaw_rate_all(self):
        """Yaw rate (about z) of every env as [N]."""
        return self._read(self.robot.get_ang())[:, 2]
```

Replace `_apply_base_control` (`scripts/scene_robot_tables.py:537-554`) with:

```python
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
```

- [ ] **Step 6: Batch `_apply_initial_pose`**

Replace `scripts/scene_robot_tables.py:516-527` with:

```python
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
```

- [ ] **Step 7: Run the math tests (sanity, no regressions)**

Run: `uv run pytest tests/test_drive_math.py -v`
Expected: all PASS (this task did not touch the math functions).

- [ ] **Step 8: Verify import still works**

Run: `uv run python -c "import sys; sys.path.insert(0,'scripts'); import scene_robot_tables; print('ok')"`
Expected: `ok`.

- [ ] **Step 9: Commit**

```bash
git add scripts/scene_robot_tables.py
git commit -m "feat: batch RobotScene controller state and per-step control"
```

---

## Task 7: Batch the public getters/setters, IK, move_ee, and teleport

Public methods squeeze to scalar when `n_envs==1` (back-compat) and pass through `[N, …]` when batched.

**Files:**
- Modify: `scripts/scene_robot_tables.py` — `set_base_velocity` (`:570-573`), `teleport_base` (`:591-607`), getters (`:609-622`), joint setters/getters (`:625-651`), IK + move_ee (`:653-688`).

- [ ] **Step 1: Batch `set_base_velocity`**

Replace `scripts/scene_robot_tables.py:570-573` with:

```python
    def set_base_velocity(self, vx, vy, wz, steps=1):
        """Set the body-frame base velocity (vx fwd, vy left, wz yaw) and advance ``steps``.

        Each of vx/vy/wz may be a scalar (same for all envs) or a length-N array.
        """
        self._twist = np.stack([
            np.broadcast_to(np.asarray(vx, dtype=float), (self.n_envs,)),
            np.broadcast_to(np.asarray(vy, dtype=float), (self.n_envs,)),
            np.broadcast_to(np.asarray(wz, dtype=float), (self.n_envs,)),
        ], axis=1)
        return self.step(steps)
```

- [ ] **Step 2: Batch the base getters**

Replace `scripts/scene_robot_tables.py:609-622` with:

```python
    def get_pos(self):
        p = self._read(self.robot.get_pos())                      # [N, 3]
        return p if self.batched else p[0]

    def get_yaw(self):
        y = self._yaw_all()                                       # [N]
        return y if self.batched else float(y[0])

    def get_yaw_rate(self):
        r = self._yaw_rate_all()                                  # [N]
        return r if self.batched else float(r[0])

    def get_base_pose(self):
        """Return ((x, y, z), yaw). Batched: ([N,3], [N]); else ([3], float)."""
        return self.get_pos(), self.get_yaw()
```

- [ ] **Step 3: Batch `teleport_base`**

Replace `scripts/scene_robot_tables.py:591-607` (the body after the docstring) with:

```python
    def teleport_base(self, x, y, yaw=None, settle=100):
        """Instantly move the base to world ``(x, y)`` (z kept) with optional ``yaw`` (rad).

        x / y / yaw may be scalars (all envs) or length-N arrays.
        """
        pos = self._read(self.robot.get_pos())                    # [N, 3]
        pos[:, 0] = np.broadcast_to(np.asarray(x, dtype=float), (self.n_envs,))
        pos[:, 1] = np.broadcast_to(np.asarray(y, dtype=float), (self.n_envs,))
        self.robot.set_pos(self._emit(pos))
        if yaw is not None:
            half = 0.5 * np.broadcast_to(np.asarray(yaw, dtype=float), (self.n_envs,))
            quat = np.stack([np.cos(half), np.zeros(self.n_envs),
                             np.zeros(self.n_envs), np.sin(half)], axis=1)  # [N, 4]
            self.robot.set_quat(self._emit(quat))
        self._twist = np.zeros((self.n_envs, 3))
        self.step(settle)
        self._heading_hold_yaw = self._yaw_all()
        return self
```

- [ ] **Step 4: Batch the joint setters/getters**

Replace `scripts/scene_robot_tables.py:625-651` with:

```python
    def set_arm(self, side, q, step=False, steps=1):
        """Set the 7 arm joint targets for ``side``. ``q`` is [7] (all envs) or [N, 7]."""
        self._arm_target[side] = self._broadcast(q, 7)
        if step:
            self.step(steps)
        return self

    def get_arm(self, side):
        q = self._read_dofs(self.arm_dofs[side])                  # [N, 7]
        return q if self.batched else q[0]

    def set_gripper(self, side, opening, step=False, steps=1):
        """Set gripper opening (m/finger). ``opening`` is a scalar (all envs) or [N]."""
        a = np.asarray(opening, dtype=float)
        if a.ndim == 0:
            self._finger_target[side] = np.full((self.n_envs, 2), float(a))
        else:
            self._finger_target[side] = np.repeat(a.reshape(self.n_envs, 1), 2, axis=1)
        if step:
            self.step(steps)
        return self

    def set_spine(self, height, step=False, steps=1):
        """Set the spine lift height (m), clamped to [0, 0.85]. Scalar or [N]."""
        h = np.clip(np.asarray(height, dtype=float), SPINE_LOWER, SPINE_UPPER)
        self._spine_target = np.broadcast_to(h, (self.n_envs,)).astype(float).copy()
        if step:
            self.step(steps)
        return self

    def get_spine(self):
        """Current spine lift height (m). Batched: [N]; else float."""
        v = self._read_dofs([self.spine_dof])[:, 0]               # [N]
        return v if self.batched else float(v[0])
```

- [ ] **Step 5: Batch the IK / TCP / move_ee methods**

Replace `scripts/scene_robot_tables.py:653-688` with:

```python
    # IK --------------------------------------------------------------
    def _tcp_of(self, side):
        """Current finger-TCP position of ``side`` (link7 origin minus tool offset).

        Batched returns [N, 3]; non-batched returns [3]."""
        link7 = self._read(self.ee_link[side].get_pos())          # [N, 3]
        tcp = link7 - np.array([0.0, 0.0, TCP_OFFSET])
        return tcp if self.batched else tcp[0]

    def get_ee_pos(self, side):
        """Current finger TCP (x, y, z) of the ``side`` arm. [N,3] batched else [3]."""
        return self._tcp_of(side)

    def ik(self, side, pos, quat=DOWN_QUAT):
        """Arm joint angles placing ``side`` finger TCP at ``pos`` with ``quat``.

        Batched: ``pos`` is [N, 3], ``quat`` [4] or [N, 4]; returns [N, 7].
        Non-batched: ``pos`` is [3]; returns [7].
        """
        offset = np.array([0.0, 0.0, TCP_OFFSET])
        if self.batched:
            target = np.broadcast_to(np.asarray(pos, dtype=float), (self.n_envs, 3)) + offset
            q = np.broadcast_to(np.asarray(quat, dtype=float), (self.n_envs, 4))
            q_full = self.robot.inverse_kinematics(
                link=self.ee_link[side], pos=target, quat=q, dofs_idx_local=self.arm_dofs[side])
            return q_full[:, self.arm_qs[side]].cpu().numpy()     # [N, 7]
        target = np.asarray(pos, dtype=float) + offset
        q_full = self.robot.inverse_kinematics(
            link=self.ee_link[side], pos=target, quat=np.asarray(quat, dtype=float),
            dofs_idx_local=self.arm_dofs[side])
        return q_full[self.arm_qs[side]].cpu().numpy()            # [7]

    def move_ee(self, side, pos, quat=DOWN_QUAT, n_waypoints=60, settle=2):
        """Move ``side`` finger TCP to ``pos`` along an interpolated straight line.

        ``pos`` is [3] (all envs) or [N, 3]; each env follows its own start->target line.
        """
        start = self._tcp_of(side)                                # [3] or [N, 3]
        if self.batched:
            start = np.atleast_2d(start)                          # [N, 3]
            target = np.broadcast_to(np.asarray(pos, dtype=float), (self.n_envs, 3))
        else:
            target = np.asarray(pos, dtype=float)
        for i in range(1, n_waypoints + 1):
            p = start + (target - start) * i / n_waypoints
            self.set_arm(side, self.ik(side, p, quat))
            self.step(settle)
        return self
```

- [ ] **Step 6: Update `reset_pose` to use batched targets**

Replace the loop body in `reset_pose` (`scripts/scene_robot_tables.py:585-587`) with:

```python
        for s in ("left", "right"):
            self._arm_target[s] = np.tile(
                np.array([ARM_HOLD[n] for n in self.arm_joint_names[s]]), (self.n_envs, 1))
            self._finger_target[s] = np.full((self.n_envs, 2), GRIPPER_OPEN)
```

- [ ] **Step 7: Run math tests + import check**

Run: `uv run pytest tests/test_drive_math.py -v && uv run python -c "import sys; sys.path.insert(0,'scripts'); import scene_robot_tables; print('ok')"`
Expected: math tests PASS, prints `ok`.

- [ ] **Step 8: Commit**

```bash
git add scripts/scene_robot_tables.py
git commit -m "feat: batch RobotScene getters/setters, IK, move_ee, teleport"
```

---

## Task 8: Manual GPU smoke test (n_envs=1 regression + n_envs=4 batched)

These exercise a real Genesis build, which needs a working backend (AMD/CUDA, or slow CPU fallback). Run on the target machine. This is the integration gate for Tasks 5-7.

**Files:**
- Create: `scripts/_smoke_parallel.py`

- [ ] **Step 1: Write the smoke script with assertions**

Create `scripts/_smoke_parallel.py`:

```python
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
```

- [ ] **Step 2: Run the smoke test on the GPU machine**

Run: `uv run python scripts/_smoke_parallel.py`
Expected (final lines):
```
[n_envs=1] OK: scalar API preserved
[n_envs=4] OK: batched shapes + per-env divergence (yaw spread=...)
SMOKE OK
```
If `spread` assertion fails, the heading-hold/`_heading_hold_yaw` re-anchoring is wrong for rotation; if shapes fail, revisit the `_read`/`_emit` adapter in Task 6.

- [ ] **Step 3: Commit**

```bash
git add scripts/_smoke_parallel.py
git commit -m "test: add manual GPU smoke test for batched RobotScene"
```

---

## Task 9: Add the CLI `--n-envs` flag and a batched `demo()`

**Files:**
- Modify: `scripts/scene_robot_tables.py` — `demo` (`:743-761`), `_parse_args` (`:765-777`), and `main` (wherever `RobotScene(...)` is constructed; find with grep below).

- [ ] **Step 1: Add the `--n-envs` argument**

In `_parse_args`, add before `return p.parse_args(argv)`:

```python
    p.add_argument("--n-envs", type=int, default=1,
                   help="Number of parallel environments (default: 1 = scalar mode).")
```

- [ ] **Step 2: Pass `n_envs` into the RobotScene constructed by `main`**

Run: `grep -n "RobotScene(" scripts/scene_robot_tables.py` to find the `main` construction site, then add `n_envs=args.n_envs` to that call, e.g.:

```python
    sim = RobotScene(headless=args.headless, save_video=args.save_video,
                     video_path=args.video_path, backend=_resolve_backend(args.backend),
                     n_envs=args.n_envs)
```

- [ ] **Step 3: Make `demo()` exercise parallelism when batched**

Replace `demo` (`scripts/scene_robot_tables.py:743-761`) with:

```python
    def demo(self):
        """Short smoke demo. Single-env: drive a path + one IK reach. Batched:
        give each env a different rotation speed so the parallelism is visible."""
        if self._record:
            self.start_recording()
        if self.batched:
            wz = np.linspace(-ANGULAR_SPEED_RADPS, ANGULAR_SPEED_RADPS, self.n_envs)
            self.set_base_velocity(LINEAR_SPEED_MPS, 0.0, 0.0, steps=150)  # all forward
            self.set_base_velocity(0.0, 0.0, wz, steps=150)                # per-env spin
            self.stop_base(steps=40)
        else:
            self.set_base_velocity(LINEAR_SPEED_MPS, 0.0, 0.0, steps=200)
            self.set_base_velocity(0.0, 0.0, ANGULAR_SPEED_RADPS, steps=120)
            self.set_base_velocity(0.0, LINEAR_SPEED_MPS, 0.0, steps=120)
            self.stop_base(steps=40)
            pos, _ = self.get_base_pose()
            self.move_ee("left", (pos[0] + 0.45, pos[1] + 0.3, 0.95), n_waypoints=40)
            self.set_arm("left", [ARM_HOLD[n] for n in self.arm_joint_names["left"]])
            self.step(60)
        if self._record:
            self.save_video()
        return self
```

- [ ] **Step 4: Verify the CLI parses (no GPU needed)**

Run: `uv run python scripts/scene_robot_tables.py --help`
Expected: help text lists `--n-envs`.

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_robot_tables.py
git commit -m "feat: add --n-envs CLI flag and batched demo()"
```

---

## Task 10: New `notebooks/parallel_demo.ipynb`

Author a notebook modeled on PhySim04: build `n_envs=9`, give each env a different rotation, record the grid video, embed it.

**Files:**
- Create: `notebooks/parallel_demo.ipynb`

- [ ] **Step 1: Create the notebook via nbformat**

Run this one-off generator (writes the notebook deterministically):

```bash
uv run python - <<'PY'
import nbformat as nbf
nb = nbf.v4.new_notebook()
md1 = """# Parallel `RobotScene` demo — N mobile dual-arm robots

This notebook runs **`n_envs` copies** of the full scene in parallel (Genesis batched
simulation) and drives each robot with a different rotation speed, like the PhySim04
parallel-simulation lab. The top-down camera frames the whole grid; the head camera
follows env 0."""
code1 = """import os
import numpy as np
from IPython.display import Video

from fr3_genesis import RobotScene

N_ENVS = 9
# env_spacing keeps the per-env scenes from overlapping in the top-down grid view.
sim = RobotScene(headless=True, save_video=True, n_envs=N_ENVS, env_spacing=(8.0, 14.0))
sim.start_recording()"""
md2 = """## Drive each environment differently

`set_base_velocity` accepts a length-N array for any of vx/vy/wz, so each env gets its
own command. Here every robot spins at a different rate."""
code2 = """# all drive forward briefly, then each spins at its own rate
sim.set_base_velocity(0.5, 0.0, 0.0, steps=150)

wz = np.linspace(-1.2, 1.2, N_ENVS)        # per-env yaw rate
sim.set_base_velocity(0.0, 0.0, wz, steps=300)
sim.stop_base(steps=40)

print("per-env yaw after spin:", np.round(sim.get_yaw(), 2))"""
md3 = """## Save and embed the grid video"""
code3 = """top_path, head_path = sim.save_video(os.path.join("videos", "parallel_demo.mp4"))
Video(top_path, embed=True, width=720)"""
nb.cells = [
    nbf.v4.new_markdown_cell(md1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(md2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(md3),
    nbf.v4.new_code_cell(code3),
]
with open("notebooks/parallel_demo.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/parallel_demo.ipynb")
PY
```

- [ ] **Step 2: Validate the notebook JSON**

Run: `uv run python -c "import json; json.load(open('notebooks/parallel_demo.ipynb')); print('valid')"`
Expected: prints `valid`.

- [ ] **Step 3: (GPU machine) execute the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/parallel_demo.ipynb`
Expected: no errors; `notebooks/videos/parallel_demo_top.mp4` is written. (Skip on a machine without a Genesis backend; note it in the PR.)

- [ ] **Step 4: Commit**

```bash
git add notebooks/parallel_demo.ipynb
git commit -m "docs: add parallel_demo notebook (n_envs=9 grid)"
```

---

## Task 11: Update README usage note

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a parallel usage snippet**

Add a short section to `README.md` (under the existing usage docs):

```markdown
### Parallel / batched simulation

Run many copies of the scene at once by passing `n_envs`:

\```python
from fr3_genesis import RobotScene
import numpy as np

sim = RobotScene(headless=True, n_envs=9, env_spacing=(8.0, 14.0))
sim.set_base_velocity(0.0, 0.0, np.linspace(-1.2, 1.2, 9), steps=300)  # per-env yaw
print(sim.get_yaw().shape)   # (9,)
\```

With `n_envs=1` (default) the API is scalar exactly as before; with `n_envs>1`
getters return a leading `[n_envs, ...]` dim and setters accept a scalar/per-dof
value (broadcast) or a full `[n_envs, ...]` array. See `notebooks/parallel_demo.ipynb`.
```

(Remove the `\` escapes — they are only here to keep this plan's code fence intact.)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document n_envs parallel simulation in README"
```

---

## Self-Review notes

- **Spec coverage:** batched build (Task 5), vectorized base controller (Tasks 2-4, 6), batched getters/setters/IK/teleport (Task 7), rendering grid framing via `env_spacing` (Tasks 5/10 — top cam already sits far above at `(0,-7,14)`; the notebook sets a wide `env_spacing` so the grid fits the existing frame, no camera-math change needed), CLI `--n-envs` + batched demo (Task 9), new notebook (Task 10), back-compat regression (Task 8 single-env check), error handling on setter shapes (`_broadcast`, Task 6). All spec sections map to a task.
- **Rendering caveat:** the spec mentions "pull the camera back/up". The existing top cam is already at `(0, -7, 14)` looking at origin; rather than recompute camera math, the demo/notebook use a wide `env_spacing` so the grid fits the current frame (simpler, YAGNI). If a future scene needs tighter framing, expose camera pos as a param then.
- **Type consistency:** `_read`→`[N,...]`, `_emit`→Genesis-shaped, `_broadcast(value,width)`→`[N,width]`, `quat_to_yaw`, `compute_drive_targets(cur[B,M],vx[B],vy[B],wz[B])` used consistently across Tasks 3-7. `_spine_target` is `[N]` everywhere (Task 6 init, set_spine, get_spine, `_apply_joint_targets` reshapes with `[:,None]`).
