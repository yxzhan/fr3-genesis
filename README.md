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

## Quick Start

Run the demo notebook on a
cloud platform — pick the one matching your GPU backend below.

### AMD GPU — AUP Learning Cloud (AUPLC)

Sign in: https://www.openhw.io/hub/

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
