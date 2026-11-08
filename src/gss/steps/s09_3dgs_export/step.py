"""Step 09: 3DGS → USDZ export via NVIDIA 3DGRUT.

Converts a trained Gaussian PLY to USDZ format for NuRec volume rendering
in Isaac Sim 5.1+. Uses 3DGRUT's ply_to_usd command in a separate conda env.

Reference: https://github.com/nv-tlabs/3dgrut
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import ThreeDGSExportConfig
from .contracts import ThreeDGSExportInput, ThreeDGSExportOutput

logger = logging.getLogger(__name__)


class ThreeDGSExportStep(
    BaseStep[ThreeDGSExportInput, ThreeDGSExportOutput, ThreeDGSExportConfig]
):
    name: ClassVar[str] = "3dgs_export"
    input_type: ClassVar = ThreeDGSExportInput
    output_type: ClassVar = ThreeDGSExportOutput
    config_type: ClassVar = ThreeDGSExportConfig

    def validate_inputs(self, inputs: ThreeDGSExportInput) -> bool:
        if not inputs.model_path.exists():
            logger.error(f"Model PLY not found: {inputs.model_path}")
            return False
        repo = self.config.threedgrut_repo
        if not repo.exists():
            logger.error(
                f"3DGRUT repo not found: {repo}. "
                "Clone with: git clone https://github.com/nv-tlabs/3dgrut.git"
            )
            return False
        return True

    def run(self, inputs: ThreeDGSExportInput) -> ThreeDGSExportOutput:
        output_dir = self.data_root / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = inputs.model_path.stem  # e.g., "point_cloud"
        usdz_path = output_dir / f"{stem}.nurec.usdz"

        # --- Run 3DGRUT ply_to_usd ---
        num_gaussians = self._run_ply_to_usd(inputs.model_path, usdz_path)

        # --- Generate NuRec manifest ---
        nurec_manifest_path = None
        if self.config.generate_manifest:
            from ._manifest import write_nurec_manifest

            nurec_manifest_path = output_dir / "scene_manifest_nurec.json"
            write_nurec_manifest(
                usdz_path=usdz_path,
                model_path=inputs.model_path,
                num_gaussians=num_gaussians,
                output_path=nurec_manifest_path,
                up_axis=self.config.up_axis,
                meters_per_unit=self.config.meters_per_unit,
            )

        logger.info(
            f"3DGS export complete: {num_gaussians} gaussians -> {usdz_path}"
        )

        return ThreeDGSExportOutput(
            usdz_path=usdz_path,
            nurec_manifest_path=nurec_manifest_path,
            num_gaussians=num_gaussians,
        )

    def _run_ply_to_usd(self, ply_path: Path, usdz_path: Path) -> int:
        """Run 3DGRUT ply_to_usd in its conda environment."""
        repo = Path(self.config.threedgrut_repo).resolve()
        conda_env = self.config.conda_env
        script = repo / "threedgrut" / "export" / "scripts" / "ply_to_usd.py"

        cmd = [
            "conda", "run", "-n", conda_env,
            "python", str(script),
            str(ply_path.resolve()),
            "--output_file", str(usdz_path.resolve()),
        ]

        logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
        logger.info(f"  cwd: {repo}")

        result = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                logger.info(f"[ply_to_usd] {line}")

        if result.returncode != 0:
            logger.error(f"[ply_to_usd] FAILED (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    logger.error(f"[ply_to_usd] {line}")
            raise RuntimeError(
                f"3DGRUT ply_to_usd failed with exit code {result.returncode}"
            )

        # Count gaussians from source PLY
        num_gaussians = self._count_ply_vertices(ply_path)
        return num_gaussians

    @staticmethod
    def _count_ply_vertices(ply_path: Path) -> int:
        """Quick header parse to count vertices in PLY."""
        with open(ply_path, "rb") as f:
            for line in f:
                decoded = line.decode("ascii", errors="ignore").strip()
                if decoded.startswith("element vertex"):
                    return int(decoded.split()[-1])
                if decoded == "end_header":
                    break
        return 0
