"""Shared pytest fixtures and markers for s07 IFC export tests."""

import pytest


def _has_ifcopenshell() -> bool:
    try:
        import ifcopenshell  # noqa: F401
        return True
    except ImportError:
        return False


needs_ifc = pytest.mark.skipif(
    not _has_ifcopenshell(), reason="ifcopenshell not installed"
)


@pytest.fixture
def ifc_context():
    """Shared IFC context — replaces _make_ctx() in all s07 test classes."""
    from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file

    return create_ifc_file()
