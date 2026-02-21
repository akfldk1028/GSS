"""Verification tests for COLMAP and gsplat installation."""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

logger = logging.getLogger(__name__)


def test_colmap_binary_available():
    """Test that COLMAP binary is available in PATH and executes."""
    try:
        result = subprocess.run(
            ["colmap", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # COLMAP --version can return non-zero exit code, so just check it runs
        assert result.returncode in [0, 1], "COLMAP should execute (exit code 0 or 1)"

        # Check that output contains version info (either in stdout or stderr)
        output = result.stdout + result.stderr
        assert "COLMAP" in output or "colmap" in output, "Output should contain COLMAP version info"

        logger.info(f"✓ COLMAP binary available: {output.strip()[:100]}")
    except FileNotFoundError:
        pytest.fail("COLMAP binary not found in PATH")
    except subprocess.TimeoutExpired:
        pytest.fail("COLMAP command timed out")
    except Exception as e:
        pytest.fail(f"COLMAP binary test failed: {e}")


def test_pycolmap_import():
    """Test that pycolmap Python package can be imported."""
    try:
        import pycolmap

        assert hasattr(pycolmap, "__version__"), "pycolmap should have __version__ attribute"
        logger.info(f"✓ pycolmap imported successfully: version {pycolmap.__version__}")
    except ImportError as e:
        pytest.fail(f"Failed to import pycolmap: {e}")
    except Exception as e:
        pytest.fail(f"pycolmap import test failed: {e}")


def test_gsplat_import():
    """Test that gsplat Python package can be imported."""
    # Use subprocess to isolate potential crashes
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import gsplat"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logger.info("✓ gsplat imported successfully")
        else:
            # Log the error for debugging
            error_msg = result.stderr or result.stdout
            pytest.fail(f"Failed to import gsplat: {error_msg[:200]}")
    except subprocess.TimeoutExpired:
        pytest.fail("gsplat import timed out")
    except Exception as e:
        pytest.fail(f"gsplat import test failed: {e}")


def test_gsplat_rasterization_import():
    """Test that gsplat.rasterization module can be imported."""
    # Use subprocess to isolate potential crashes
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from gsplat import rasterization; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and "OK" in result.stdout:
            logger.info("✓ gsplat.rasterization imported successfully")
        else:
            error_msg = result.stderr or result.stdout
            pytest.fail(f"Failed to import gsplat.rasterization: {error_msg[:200]}")
    except subprocess.TimeoutExpired:
        pytest.fail("gsplat.rasterization import timed out")
    except Exception as e:
        pytest.fail(f"gsplat.rasterization import test failed: {e}")


def test_gsplat_cuda_compatibility():
    """Test that gsplat works with CUDA (if available)."""
    # Use subprocess to isolate potential crashes
    check_script = """
import torch
import gsplat
cuda_available = torch.cuda.is_available()
print(f"CUDA:{cuda_available}")
if cuda_available:
    from gsplat import rasterization
    print("OK")
else:
    print("SKIP")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", check_script],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            if "CUDA:True" in output and "OK" in output:
                logger.info("✓ gsplat CUDA compatibility verified")
            elif "SKIP" in output:
                logger.warning("CUDA not available - skipping CUDA compatibility check")
                pytest.skip("CUDA not available on this system")
            else:
                logger.info(f"CUDA check result: {output}")
        else:
            error_msg = result.stderr or result.stdout
            pytest.fail(f"Failed to verify gsplat CUDA compatibility: {error_msg[:200]}")
    except subprocess.TimeoutExpired:
        pytest.fail("gsplat CUDA compatibility check timed out")
    except Exception as e:
        pytest.fail(f"gsplat CUDA compatibility test failed: {e}")


@pytest.mark.e2e
def test_colmap_gsplat_integration():
    """
    Integration test verifying all COLMAP and gsplat components work together.

    This test ensures that:
    1. COLMAP CLI is accessible for S02 step
    2. pycolmap Python bindings work for programmatic access
    3. gsplat imports correctly for S03 step
    4. gsplat rasterization module is available for GPU rendering
    """
    logger.info("Running COLMAP & gsplat integration test")

    # Test 1: COLMAP binary
    try:
        result = subprocess.run(
            ["colmap", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode in [0, 1], "COLMAP binary should execute"
        logger.info("✓ COLMAP binary verified")
    except Exception as e:
        pytest.fail(f"COLMAP binary check failed: {e}")

    # Test 2: pycolmap import
    try:
        import pycolmap
        logger.info(f"✓ pycolmap {pycolmap.__version__} verified")
    except ImportError as e:
        pytest.fail(f"pycolmap import failed: {e}")

    # Test 3: gsplat import
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import gsplat"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("✓ gsplat verified")
        else:
            pytest.fail(f"gsplat import failed: {result.stderr[:200]}")
    except Exception as e:
        pytest.fail(f"gsplat import failed: {e}")

    # Test 4: gsplat rasterization
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from gsplat import rasterization; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            logger.info("✓ gsplat.rasterization verified")
        else:
            pytest.fail(f"gsplat.rasterization import failed: {result.stderr[:200]}")
    except Exception as e:
        pytest.fail(f"gsplat.rasterization import failed: {e}")

    logger.info("✓ All COLMAP & gsplat integration checks passed")
