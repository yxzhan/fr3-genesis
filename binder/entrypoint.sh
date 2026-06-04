#!/bin/bash

# Source ROS2
# source ${ROS_PATH}/setup.bash

# Add other startup programs here

# The following line will allow the binderhub start Jupyterlab, should be at the end of the entrypoint.
# Do not modify it!
exec "$@"