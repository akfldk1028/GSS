"""Shared fixtures for integration tests using real data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Real data directory (from actual pipeline run)
REAL_DATA_ROOT = Path("data")
REAL_S06_DIR = REAL_DATA_ROOT / "interim" / "s06_planes"


def _has_real_data() -> bool:
    """Check if real s06 pipeline output exists."""
    return (
        (REAL_S06_DIR / "planes.json").exists()
        and (REAL_S06_DIR / "boundaries.json").exists()
    )


requires_real_data = pytest.mark.skipif(
    not _has_real_data(),
    reason="Real s06 data not found at data/interim/s06_planes/",
)


@pytest.fixture
def real_planes() -> list[dict]:
    """Load real planes.json."""
    with open(REAL_S06_DIR / "planes.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def real_planes_file() -> Path:
    return REAL_S06_DIR / "planes.json"


@pytest.fixture
def real_boundaries_file() -> Path:
    return REAL_S06_DIR / "boundaries.json"


@pytest.fixture
def real_manhattan_rotation():
    """Load Manhattan rotation if available."""
    path = REAL_S06_DIR / "manhattan_alignment.json"
    if not path.exists():
        return None
    import numpy as np
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return np.asarray(data["manhattan_rotation"], dtype=float)
