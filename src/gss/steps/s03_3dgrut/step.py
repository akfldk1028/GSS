"""Step 03: 3DGRUT training (NVIDIA 3D Gaussian Ray Tracing).

Invokes 3DGRUT as a subprocess using Hydra config overrides.
Exports PLY (for downstream BIM pipeline) and optionally USDZ
(for direct Omniverse/ARKit loading).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import S03_3dgrutConfig
from .contracts import S03_3dgrutInput, S03_3dgrutOutput

logger = logging.getLogger(__name__)


def _has_threedgrut(grut_path: Path) -> bool:
    """Check if 3DGRUT repo is available and has the training script."""
    train_script = grut_path / "threedgrut" / "scripts" / "train.py"
    if not train_script.exists():
        # Fallback: check for main entry point
        alt_script = grut_path / "train.py"
        return alt_script.exists()
    return True


def _find_train_script(grut_path: Path) -> Path:
    """Locate the 3DGRUT training entry point."""
    candidates = [
        grut_path / "threedgrut" / "scripts" / "train.py",
        grut_path / "train.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"3DGRUT training script not found in {grut_path}. "
        "Expected: threedgrut/scripts/train.py or train.py"
    )


def _find_export_script(grut_path: Path) -> Path | None:
    """Locate the 3DGRUT export script (optional)."""
    candidates = [
        grut_path / "threedgrut" / "scripts" / "export.py",
        grut_path / "export.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _count_ply_points(ply_path: Path) -> int:
    """Read the vertex count from a PLY file header."""
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(ply_path))
        return len(ply["vertex"])
    except Exception:
        return 0


class ThreeDGRUTStep(
    BaseStep[S03_3dgrutInput, S03_3dgrutOutput, S03_3dgrutConfig]
):
    name: ClassVar[str] = "3dgrut"
    input_type: ClassVar = S03_3dgrutInput
    output_type: ClassVar = S03_3dgrutOutput
    config_type: ClassVar = S03_3dgrutConfig

    def validate_inputs(self, inputs: S03_3dgrutInput) -> bool:
        if not inputs.frames_dir.exists():
            logger.error("Frames directory not found: %s", inputs.frames_dir)
            return False
        if not inputs.sparse_dir.exists():
            logger.error("COLMAP sparse dir not found: %s", inputs.sparse_dir)
            return False

        grut_path = Path(self.config.threedgrut_path)
        if not _has_threedgrut(grut_path):
            logger.error(
                "3DGRUT not found at %s — clone it first: "
                "git clone https://github.com/nv-tlabs/3dgrut.git clone/3dgrut",
                grut_path,
            )
            return False

        if self.config.method not in ("3dgut", "3dgrt"):
            logger.error("Invalid method: %s (must be '3dgut' or '3dgrt')", self.config.method)
            return False

        return True

    def run(self, inputs: S03_3dgrutInput) -> S03_3dgrutOutput:
        output_dir = self.data_root / "interim" / "s03_3dgrut"
        output_dir.mkdir(parents=True, exist_ok=True)

        grut_path = Path(self.config.threedgrut_path).resolve()
        train_script = _find_train_script(grut_path)

        # ── 1. Training via subprocess ──────────────────────────────
        # 3DGRUT uses Hydra; we pass COLMAP path + output dir as overrides
        cmd = [
            sys.executable,
            str(train_script),
            "--config-name", self.config.config_name,
            f"dataset.path={inputs.sparse_dir.parent.resolve().as_posix()}",
            f"trainer.out_dir={output_dir.resolve().as_posix()}",
            f"trainer.num_iterations={self.config.num_iterations}",
        ]
        # Append any extra Hydra overrides
        cmd.extend(self.config.extra_args)

        logger.info("Starting 3DGRUT training: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(grut_path),
            capture_output=True,
            text=True,
            timeout=7200,  # 2h max
        )
        if result.returncode != 0:
            logger.error("3DGRUT training stderr:\n%s", result.stderr[-2000:])
            raise RuntimeError(
                f"3DGRUT training failed (exit code {result.returncode}). "
                f"See log above for details."
            )
        logger.info("3DGRUT training completed")

        # ── 2. Locate checkpoint ────────────────────────────────────
        checkpoint_path = _find_checkpoint(output_dir)
        logger.info("Checkpoint: %s", checkpoint_path)

        # ── 3. Export PLY ───────────────────────────────────────────
        ply_path = output_dir / "point_cloud.ply"
        if self.config.export_ply:
            self._export_ply(grut_path, checkpoint_path, ply_path)

        # ── 4. Export USDZ (optional) ──────────────────────────────
        usdz_path = None
        if self.config.export_usdz:
            usdz_path = self._export_usdz(grut_path, checkpoint_path, output_dir)

        # ── 5. Count gaussians ─────────────────────────────────────
        num_gaussians = _count_ply_points(ply_path) if ply_path.exists() else 0

        return S03_3dgrutOutput(
            model_path=ply_path,
            checkpoint_path=checkpoint_path,
            usdz_path=usdz_path,
            num_gaussians=num_gaussians,
            training_iterations=self.config.num_iterations,
        )

    def _export_ply(self, grut_path: Path, checkpoint_path: Path, ply_path: Path) -> None:
        """Export trained model to PLY for downstream pipeline compatibility."""
        export_script = _find_export_script(grut_path)
        if export_script is None:
            logger.warning(
                "3DGRUT export script not found — skipping PLY export. "
                "The checkpoint can still be used for gRPC rendering."
            )
            return

        cmd = [
            sys.executable,
            str(export_script),
            f"--checkpoint={checkpoint_path}",
            f"--output={ply_path}",
            "--format=ply",
        ]
        logger.info("Exporting PLY: %s", ply_path)
        result = subprocess.run(
            cmd, cwd=str(grut_path), capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.warning("PLY export failed: %s", result.stderr[-500:])

    def _export_usdz(
        self, grut_path: Path, checkpoint_path: Path, output_dir: Path
    ) -> Path | None:
        """Export trained model to USDZ via 3DGRUT's built-in exporter."""
        export_script = _find_export_script(grut_path)
        if export_script is None:
            logger.warning("3DGRUT export script not found — skipping USDZ export.")
            return None

        usdz_path = output_dir / "model.usdz"
        cmd = [
            sys.executable,
            str(export_script),
            f"--checkpoint={checkpoint_path}",
            f"--output={usdz_path}",
            "--format=usdz",
        ]
        logger.info("Exporting USDZ: %s", usdz_path)
        result = subprocess.run(
            cmd, cwd=str(grut_path), capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.warning("USDZ export failed: %s", result.stderr[-500:])
            return None

        return usdz_path if usdz_path.exists() else None


def _find_checkpoint(output_dir: Path) -> Path:
    """Locate the latest 3DGRUT checkpoint in the output directory."""
    # 3DGRUT saves checkpoints as .pt or .ckpt files
    patterns = ["**/*.pt", "**/*.ckpt", "**/*.pth"]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(output_dir.glob(pat))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {output_dir}. "
            "3DGRUT training may have failed silently."
        )

    # Return most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)
