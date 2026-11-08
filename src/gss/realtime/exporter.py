"""PLY → USDZ export via 3DGRUT for NuRec rendering.

Wraps the 3DGRUT ply_to_usd script in a subprocess call,
using the same approach as s09 but standalone.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .config import ExporterConfig

logger = logging.getLogger(__name__)


class USDZExporter:
    """Converts Gaussian PLY to NuRec USDZ."""

    def __init__(self, config: ExporterConfig) -> None:
        self._config = config
        self._repo = Path(config.threedgrut_repo).resolve()
        self._script = self._repo / "threedgrut" / "export" / "scripts" / "ply_to_usd.py"

    def validate(self) -> bool:
        """Check that 3DGRUT is available."""
        if not self._repo.exists():
            logger.error(f"3DGRUT repo not found: {self._repo}")
            return False
        if not self._script.exists():
            logger.error(f"ply_to_usd.py not found: {self._script}")
            return False
        return True

    def export(self, ply_path: Path, output_path: Path) -> Path:
        """Convert PLY to USDZ. Returns output path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", self._config.conda_env,
            "python", str(self._script),
            str(ply_path.resolve()),
            "--output_file", str(output_path.resolve()),
        ]

        logger.info(f"Exporting: {ply_path.name} -> {output_path.name}")
        result = subprocess.run(cmd, cwd=str(self._repo), capture_output=True, text=True)

        if result.returncode != 0:
            stderr_tail = "\n".join(result.stderr.strip().split("\n")[-10:])
            logger.error(f"ply_to_usd failed:\n{stderr_tail}")
            raise RuntimeError(f"3DGRUT export failed (exit {result.returncode})")

        logger.info(f"USDZ exported: {output_path}")
        return output_path

    def export_with_manifest(
        self, ply_path: Path, output_dir: Path
    ) -> tuple[Path, Path]:
        """Export USDZ + generate NuRec manifest. Returns (usdz_path, manifest_path)."""
        usdz_path = output_dir / f"{ply_path.stem}.nurec.usdz"
        self.export(ply_path, usdz_path)

        manifest_path = output_dir / "scene_manifest_nurec.json"
        if self._config.generate_manifest:
            self._write_manifest(usdz_path, ply_path, manifest_path)

        return usdz_path, manifest_path

    def _write_manifest(
        self, usdz_path: Path, ply_path: Path, output_path: Path
    ) -> None:
        num_gaussians = _count_ply_vertices(ply_path)
        manifest = {
            "format": "nurec",
            "version": "1.0",
            "usdz_path": str(usdz_path),
            "source_ply": str(ply_path),
            "num_gaussians": num_gaussians,
            "stage": {
                "up_axis": self._config.up_axis,
                "meters_per_unit": 1.0,
            },
            "rendering": {
                "mode": "volume",
                "engine": "nurec",
                "requires_isaac_sim": "5.1.0",
            },
            "spawn_points": [
                {"name": "default", "position": [0, 0, 0], "orientation": [1, 0, 0, 0]},
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written: {output_path}")


def _count_ply_vertices(ply_path: Path) -> int:
    with open(ply_path, "rb") as f:
        for line in f:
            decoded = line.decode("ascii", errors="ignore").strip()
            if decoded.startswith("element vertex"):
                return int(decoded.split()[-1])
            if decoded == "end_header":
                break
    return 0
