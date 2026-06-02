# fr3-genesis

[中文](README.zh-CN.md)

Genesis examples for the `mobile_fr3_duo` dual-arm mobile robot: pick-and-place,
keyboard teleoperation, and scene display.

Scripts under `scripts/`:

- `fr3_genesis.py` — pick-and-place demo with the left arm + gripper (records a video to `videos/`).
- `keyboard_control.py` — WASD/arrow-key teleop of the mobile base + both arms locked in pose.
- `scene_robot_keyboard.py` — full scene (tables/letters/tableware) + a controllable robot.
- `scene_robot_tables.py` — static scene display, no manipulation task.

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
uv run python scripts/fr3_genesis.py
uv run python scripts/keyboard_control.py
```

> Note: the pick-and-place demo (`fr3_genesis.py`) uses the `gs.gpu` backend by
> default. For a CPU-only install, change `gs.init(backend=gs.gpu)` to
> `gs.init(backend=gs.cpu)` in the script.
