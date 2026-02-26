"""Module F: Opening detection in walls (Phase 2 - stub).

Detect doors and windows by analyzing gaps in wall point distributions.
This module is disabled by default and will be implemented in Phase 2.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_openings(
    planes: list[dict],
    walls: list[dict],
    surface_points_path: str | None = None,
) -> list[dict]:
    """Detect openings (doors, windows) in wall planes.

    Phase 2 implementation. Currently returns empty list.

    Args:
        planes: plane dicts in Manhattan space.
        walls: wall info dicts from wall_thickness module.
        surface_points_path: optional path to surface point cloud.

    Returns:
        list of opening dicts (empty for now).
    """
    logger.info("Opening detection: Phase 2 stub, skipping")
    return []
