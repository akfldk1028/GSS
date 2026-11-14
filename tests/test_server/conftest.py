"""Server test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture()
def client():
    """FastAPI TestClient with lifespan (starts workers)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def dummy_ply(tmp_path):
    """Create a minimal valid PLY file for upload tests."""
    ply_path = tmp_path / "test.ply"
    ply_content = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
    )
    ply_path.write_text(ply_content)
    return ply_path
