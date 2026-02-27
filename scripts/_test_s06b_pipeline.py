"""Test: verify s06b integrates correctly in the pipeline runner."""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from gss.core.pipeline_runner import import_step_class, load_step_config

# 1. Dynamic import works
step_cls = import_step_class("gss.steps.s06b_plane_regularization")
print(f"Step class: {step_cls.__name__}")
print(f"Input type: {step_cls.input_type.__name__}")
print(f"Output type: {step_cls.output_type.__name__}")
print(f"Config type: {step_cls.config_type.__name__}")

# 2. Config loading works
config = load_step_config(
    Path("configs/steps/s06b_plane_regularization.yaml"),
    step_cls.config_type,
)
print(f"Config loaded: {config.model_dump()}")

# 3. Input from s06 output (simulated pipeline model_dump merge)
from gss.steps.s06_plane_extraction.contracts import PlaneExtractionOutput
s06_output = PlaneExtractionOutput(
    planes_file=Path("data/interim/s06_planes/planes.json"),
    boundaries_file=Path("data/interim/s06_planes/boundaries.json"),
    num_planes=22,
    num_walls=3,
    num_floors=1,
    num_ceilings=2,
)
input_data = s06_output.model_dump()
s06b_input = step_cls.input_type(**input_data)
print(f"s06b input from s06 output: OK (planes_file={s06b_input.planes_file})")

# 4. Output compatibility with s07 input
from gss.steps.s07_ifc_export.contracts import IfcExportInput
from gss.steps.s06b_plane_regularization.contracts import PlaneRegularizationOutput
s06b_output = PlaneRegularizationOutput(
    planes_file=Path("data/interim/s06b_plane_regularization/planes.json"),
    boundaries_file=Path("data/interim/s06b_plane_regularization/boundaries.json"),
    walls_file=Path("data/interim/s06b_plane_regularization/walls.json"),
    num_walls=3,
)
output_data = s06b_output.model_dump()
# s07 now requires walls_file (mandatory), planes_file is optional fallback
s07_input = IfcExportInput(
    walls_file=output_data["walls_file"],
    planes_file=output_data.get("planes_file"),
    boundaries_file=output_data.get("boundaries_file"),
    spaces_file=output_data.get("spaces_file"),
)
print(f"s07 input from s06b output: OK (walls_file={s07_input.walls_file})")

print("\nAll pipeline integration checks passed!")
