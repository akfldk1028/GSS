"""Test PlanarGS step with Replica room0 data."""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gss.steps.s03_planargs import PlanarGSStep, PlanarGSInput, PlanarGSConfig

data_root = Path("data")
config = PlanarGSConfig(
    planargs_repo=Path("clone/PlanarGS"),
    conda_env="planargs",
    group_size=25,
    iterations=30000,
)
step = PlanarGSStep(config=config, data_root=data_root)
out = step.execute(PlanarGSInput(
    frames_dir=Path("data/interim/s01_frames"),
    sparse_dir=Path("data/interim/s02_colmap/sparse/0"),
))
print(f"\nResult: {out.num_surface_points} points")
print(f"  surface_points: {out.surface_points_path}")
print(f"  mesh: {out.mesh_path}")
print(f"  metadata: {out.metadata_path}")
