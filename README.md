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

A worked example of the `RobotScene` API lives in `notebooks/robot_scene_demo.ipynb`.

## Installation

Dependencies: `genesis-world`, `pynput`, `numpy`, plus **one PyTorch variant
chosen for your hardware** (CPU / NVIDIA CUDA / AMD ROCm). This project uses
[uv](https://docs.astral.sh/uv/) to manage the virtual environment and
dependencies.

### 1. Install uv (one-time)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or see the official docs: https://docs.astral.sh/uv/getting-started/installation/

### 2. Create the virtual environment and install dependencies

`uv` creates the virtual environment at `./.venv` in the repo root. **Pick one
extra for your hardware:**

```bash
uv venv                      # create the virtual environment at ./.venv

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

### 3. Run

```bash
# reusable scene library — runs a short demo; --save-video writes a top-down + head-camera mp4
uv run python scripts/scene_robot_tables.py --headless --save-video

# keyboard teleop (needs a display; falls back to a scripted video when headless)
uv run python scripts/scene_robot_keyboard.py
```

> Note: both scripts auto-select the Genesis backend (CUDA → ROCm → Metal → CPU)
> based on your hardware, so no manual backend edit is needed.

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
