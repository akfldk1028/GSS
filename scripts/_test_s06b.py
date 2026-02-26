"""Quick test: run s06b on existing s06 output."""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from gss.steps.s06b_plane_regularization.step import PlaneRegularizationStep
from gss.steps.s06b_plane_regularization.contracts import PlaneRegularizationInput
from gss.steps.s06b_plane_regularization.config import PlaneRegularizationConfig

config = PlaneRegularizationConfig()
step = PlaneRegularizationStep(config=config, data_root=Path("data"))
inp = PlaneRegularizationInput(
    planes_file=Path("data/interim/s06_planes/planes.json"),
    boundaries_file=Path("data/interim/s06_planes/boundaries.json"),
)
output = step.execute(inp)

print()
print("=== Output ===")
print(f"planes_file: {output.planes_file}")
print(f"boundaries_file: {output.boundaries_file}")
print(f"walls_file: {output.walls_file}")
print(f"spaces_file: {output.spaces_file}")
print(f"num_walls: {output.num_walls}")
print(f"num_spaces: {output.num_spaces}")
