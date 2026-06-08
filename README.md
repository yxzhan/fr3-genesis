# fr3-genesis

Genesis examples for the `mobile_fr3_duo` dual-arm mobile robot: a reusable
scene/robot library and keyboard teleoperation.

Scripts under `scripts/`:

- `scene_robot_tables.py` — reusable `RobotScene` library. Builds the full scene
  (tables / letters / tableware) with a mobile dual-arm robot you can drive,
  joint-control, and move via IK; supports headless rendering and head-camera +
  top-down video recording. Import it (see `notebooks/robot_scene_demo.ipynb`) or
  run it directly for a short demo.
- `scene_robot_keyboard.py` — full scene + keyboard teleop of the mobile base
  (arrow keys to translate, `,` / `.` to rotate); records a scripted video when headless.

## Demo

Recorded from the pick-and-place walkthrough in
[`notebooks/robot_scene_tables.ipynb`](notebooks/robot_scene_tables.ipynb).

| Top-down camera | Robot head camera |
| --- | --- |
| <video src="https://github.com/yxzhan/fr3-genesis/raw/main/docs/robot_scene_tables_top.mp4" controls autoplay loop muted playsinline width="500"></video> | <video src="https://github.com/yxzhan/fr3-genesis/raw/main/docs/robot_scene_tables_head.mp4" controls autoplay loop muted playsinline width="500"></video> |

## Quick Start

Run the demo notebook on a
cloud platform — pick the one matching your GPU backend below.

### AMD GPU — AUP Learning Cloud (AUPLC)

Sign in: https://tpe.aupcloud.io/hub/home

1. Start the **"Genesis Physical Simulation Course"** server.
2. Open a **Terminal**.
3. Expose your persistent home directory in the file browser:
   ```bash
   ln -s ~
   ```
   `/home/jovyan` (`~`) is the only directory that persists across server
   restarts, but JupyterLab does not show it by default. This command creates a
   `jovyan` symlink in the current working directory so you can browse it. The
   symlink itself is not persistent — **re-run it each time you start the server.**
4. Clone this repository into your home directory so your work survives restarts:
   ```bash
   git clone https://github.com/yxzhan/fr3-genesis.git ~/fr3-genesis
   ```
5. Open and run [`notebooks/robot_scene_tables.ipynb`](notebooks/robot_scene_tables.ipynb).

> Headless only — no display, so keep `HEADLESS = True` in the notebook.

### CPU — AICOR Virtual Research Building

https://binder.intel4coro.de/v2/gh/yxzhan/fr3-genesis/main?urlpath=lab/tree/notebooks/robot_scene_tables.ipynb

> Supports the interactive Genesis viewer (set `HEADLESS = False` in the
> notebook), but runs on CPU so it is noticeably slower. Before running in
> non-headless mode, open the **"Desktop"** tab first so the virtual display is
> initialized.

### NVIDIA GPU — Google Colab

https://colab.research.google.com/github/yxzhan/fr3-genesis/blob/main/notebooks/robot_scene_tables.ipynb

1. Set the runtime to GPU: **Runtime → Change runtime type → GPU**.
2. Run the first (setup) cell. When dependency installation finishes, Colab
   prompts you to **restart the kernel** — do so, then run the notebook from the
   top.

> Headless only — no display, so keep `HEADLESS = True` in the notebook.


## Known issues on AUPLC (AMD GPU)

Gotchas specific to the AUP Learning Cloud / AMD ROCm environment, and how this
repo works around them:

1. **No `sudo` in the container.** The single-user container is unprivileged, so
   you cannot `apt install` ROS 2 (or other system packages). Rely on what the
   course image already provides and install Python dependencies into the base environment.

2. **Pin `genesis-world` to 0.4.6.** Upgrading to `genesis-world` 1.0.0 crashes
   the Jupyter kernel on this platform. The dependency is pinned to `==0.4.6` in
   `pyproject.toml`.

3. **CoACD convex decomposition crashes the kernel on AMD.** The first time the
   scene is built, Genesis runs CoACD convex decomposition on the collision
   meshes, and that call crashes the kernel under the ROCm/AMD backend. As a
   workaround the decomposition is precomputed on a CUDA machine and the result
   is bundled in `assets/genesis_cache`; on startup `RobotScene` seeds
   `~/.cache/genesis` from it (`_restore_genesis_cache` in
   `scripts/scene_robot_tables.py`) so Genesis never decomposes anything at
   runtime. Keep the bundled cache in sync with the pinned Genesis version — the
   cache key includes `gs.__version__`.

4. **Base driving jitters violently → use 64-bit precision.** When driving the
   FR3 mobile base, the robot shakes badly on AMD, almost certainly due to
   floating-point/solver numerical differences in the backend. It is fixed by
   initializing Genesis with 64-bit precision on the AMD backend (`precision="64"`
   when `backend is gs.amdgpu`; see `_ensure_gs_init` in
   `scripts/scene_robot_tables.py`). Other backends keep 32-bit.


## Installation

This project uses
[uv](https://docs.astral.sh/uv/) to manage the virtual environment and
dependencies.

### 1. Install uv (one-time)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or see the official docs: https://docs.astral.sh/uv/getting-started/installation/

### 2. Create the virtual environment and install dependencies

Dependencies: `genesis-world` plus **one PyTorch variant
chosen for your hardware** (CPU / NVIDIA CUDA / AMD ROCm).

`uv` creates the virtual environment at `./.venv` in the repo root. **Pick one
extra for your hardware:**

```bash
# —— choose one ——
uv sync --extra cpu          # no GPU / CI: CPU build of PyTorch
uv sync --extra cu126        # NVIDIA GPU: CUDA 12.6 build of PyTorch
uv sync --extra rocm         # AMD GPU (Linux only): ROCm 7.2 build of PyTorch
```

- `cpu` / `cu126` / `rocm` are mutually exclusive — install only one at a time;
  to switch variants, just re-run the matching `uv sync --extra ...`.
- NVIDIA defaults to `cu126` (CUDA 12.6); for a different CUDA version, change
  the `pytorch-cu126` index URL in `pyproject.toml` accordingly (e.g. `whl/cu128`).
- ROCm is Linux-only.

### 3. Usage

See [notebooks/robot_scene_tables.ipynb](notebooks/robot_scene_tables.ipynb).

```bash
# reusable scene library — runs a short demo; --save-video writes a top-down + head-camera mp4
uv run python scripts/scene_robot_tables.py --headless --save-video

# keyboard teleop (needs a display; falls back to a scripted video when headless)
uv run python scripts/scene_robot_keyboard.py
```

### Parallel / batched simulation

Run many copies of the scene at once by passing `n_envs`:

```python
from fr3_genesis import RobotScene
import numpy as np

sim = RobotScene(headless=True, n_envs=9, env_spacing=(8.0, 14.0))
sim.set_base_velocity(0.0, 0.0, np.linspace(-1.2, 1.2, 9), steps=300)  # per-env yaw
print(sim.get_yaw().shape)   # (9,)
```

With `n_envs=1` (default) the API is scalar exactly as before; with `n_envs>1`
getters return a leading `[n_envs, ...]` dim and setters accept a scalar/per-dof
value (broadcast) or a full `[n_envs, ...]` array. See `notebooks/parallel_demo.ipynb`.

## Install as a package (`fr3_genesis`)

The project is also a pip-installable package that bundles the bundled `assets/`
(URDF / MJCF / convex-decomposition cache), exposing `from fr3_genesis import RobotScene`.

### From the git repository, without dependencies

Install only this project's code + assets, **without** pulling in `genesis-world`,
`torch`, etc. (use this when those dependencies are already present in your
environment):

```bash
pip install --no-deps git+https://github.com/yxzhan/fr3-genesis.git
# or pin a branch / tag / commit:
pip install --no-deps git+https://github.com/yxzhan/fr3-genesis.git@main
# with uv:
uv pip install --no-deps git+https://github.com/yxzhan/fr3-genesis.git
```

Then:

```python
from fr3_genesis import RobotScene
sim = RobotScene(headless=True)
```

Notes:
- `--no-deps` skips dependency installation but **still builds the package**, so
  `assets/` (incl. the large URDF meshes) are downloaded and installed.
- At runtime `genesis` and the other dependencies must already be importable in
  your environment, otherwise `import fr3_genesis` raises `ModuleNotFoundError`.
  Drop `--no-deps` (and add a torch extra) to install everything:
  `pip install "fr3-genesis[cpu] @ git+https://github.com/yxzhan/fr3-genesis.git"`.
