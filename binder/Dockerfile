# Dockerfile of the base image: https://github.com/IntEL4CoRo/jupyter-ros2/blob/main/Dockerfile
FROM intel4coro/jupyter-ros2:jazzy-py3.12

USER root
ENV PATH=$HOME/.local/bin:$PATH

# Set Working Directory
ENV REPO_DIR=/home/repo
RUN mkdir -p ${REPO_DIR}
WORKDIR ${REPO_DIR}

# Franka robot dependecies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-realsense2-description \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-realsense2-camera \
    ros-${ROS_DISTRO}-sick-safetyscanners2 \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-rqt-robot-steering \
    && rm -rf /var/lib/apt/lists/*

# Upgrade vscode-server
RUN mamba update -y conda-forge::code-server
RUN code-server --install-extension mhutchie.git-graph
RUN code-server --install-extension lichenxi.sysmonitor
RUN curl -LO https://github.com/yxzhan/vscode_remote_desktop/releases/download/v0.1.0/vscode-remote-desktop-0.1.0.vsix && \
    code-server --install-extension vscode-remote-desktop-0.1.0.vsix && \
    rm vscode-remote-desktop-0.1.0.vsix

# Install Claude code
RUN curl -fsSL https://claude.ai/install.sh | bash

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy content to the image
COPY . ${REPO_DIR}/
RUN git config --global --add safe.directory ${REPO_DIR}

# Build the project venv (.venv) and register it as a Jupyter kernel.
# UV_EXTRA selects the PyTorch build: cpu | cu126 | rocm (default: cpu).
ARG UV_EXTRA=cpu
RUN uv sync --extra ${UV_EXTRA} && \
    uv pip install ipykernel && \
    uv run python -m ipykernel install --user \
        --name genesis --display-name "Python (Genesis)"

# Set JupyterLab default directory
WORKDIR ${REPO_DIR}
# Set VSCODE default directory
ENV CODE_WORKING_DIRECTORY=${REPO_DIR}

# The entrypoint of the docker image
COPY binder/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
