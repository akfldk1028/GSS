"""Safe subprocess runner for external tools (COLMAP, ffmpeg, etc.)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 3600,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run an external command with logging and error handling."""
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    if result.stdout:
        logger.debug(f"stdout: {result.stdout[-500:]}")
    if result.stderr:
        logger.debug(f"stderr: {result.stderr[-500:]}")

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd_str, result.stdout, result.stderr
        )
    return result
