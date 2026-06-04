"""fr3_genesis -- pip-installable Genesis scene for the mobile FR3 dual-arm robot.

Usage::

    from fr3_genesis import RobotScene
    sim = RobotScene(headless=True)
    sim.set_base_velocity(0.5, 0.0, 0.0, steps=200)

The real implementation lives in ``scene_robot_tables``. In a built wheel that module
(and the ``assets/`` tree) is bundled into this package; when running from the source
checkout it still lives under ``<repo>/scripts``, so fall back to importing it there.
"""

try:
    # Installed wheel: scene_robot_tables.py is bundled inside this package.
    from .scene_robot_tables import RobotScene, main
except ImportError:  # pragma: no cover - source-checkout / editable fallback
    import os
    import sys

    _scripts = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from scene_robot_tables import RobotScene, main

__all__ = ["RobotScene", "main"]
