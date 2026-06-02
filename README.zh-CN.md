# fr3-genesis

[English](README.md)

Genesis 版 `mobile_fr3_duo` 双臂移动机器人示例：抓取放置、键盘遥控、场景展示。

`scripts/` 下的脚本：

- `fr3_genesis.py` — 左臂 + 夹爪的抓取-放置 demo（录视频到 `videos/`）。
- `keyboard_control.py` — WASD/方向键 遥控移动底盘 + 双臂锁姿态。
- `scene_robot_keyboard.py` — 完整场景（桌子/字母/餐具）+ 可控机器人。
- `scene_robot_tables.py` — 静态场景展示，无抓取任务。

## 安装

依赖：`genesis-world`、`pynput`、`numpy`，以及 **按你的硬件选一个变体的 PyTorch**
（CPU / NVIDIA CUDA / AMD ROCm）。本项目用 [uv](https://docs.astral.sh/uv/) 管理虚拟
环境和依赖。

### 1. 安装 uv（一次性）

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或参考官方文档：https://docs.astral.sh/uv/getting-started/installation/

### 2. 在当前目录创建虚拟环境并安装依赖

`uv` 会把虚拟环境创建在仓库根目录的 `./.venv` 下。**根据硬件选择一个 extra**：

```bash
uv venv                      # 在 ./.venv 创建虚拟环境

# —— 三选一 ——
uv sync --extra cpu          # 无 GPU / CI：CPU 版 PyTorch
uv sync --extra cu126        # NVIDIA 显卡：CUDA 12.6 版 PyTorch
uv sync --extra rocm         # AMD 显卡（仅 Linux）：ROCm 7.2 版 PyTorch
```

- `cpu` / `cu126` / `rocm` 三者互斥，一次只能装一个；切换变体重新跑对应的
  `uv sync --extra ...` 即可。
- 默认 NVIDIA 用 `cu126`（CUDA 12.6）；如需其它 CUDA 版本，把 `pyproject.toml`
  里的 `pytorch-cu126` 索引 URL 改成对应版本（如 `whl/cu128`）即可。
- ROCm 仅 Linux 可用。

### 3. 运行

```bash
uv run python scripts/fr3_genesis.py
uv run python scripts/keyboard_control.py
```

> 提示：抓取 demo（`fr3_genesis.py`）默认用 `gs.gpu` 后端。纯 CPU 安装时，把脚本里
> `gs.init(backend=gs.gpu)` 改成 `gs.init(backend=gs.cpu)`。
