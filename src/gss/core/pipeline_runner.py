"""Pipeline orchestrator: reads pipeline.yaml and executes steps in order."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

from .contracts import PipelineConfig

logger = logging.getLogger(__name__)


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    """Load and validate pipeline.yaml."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)


def load_step_config(config_path: Path, config_class: type[BaseModel]) -> BaseModel:
    """Load a step-specific YAML config into its Pydantic model."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return config_class(**raw)


def import_step_class(module_path: str):
    """Dynamically import a step class from its module path.

    Expects module_path like 'gss.steps.s01_extract_frames'
    and looks for a class ending in 'Step' in that module's step.py.
    """
    step_module = importlib.import_module(f"{module_path}.step")
    for attr_name in dir(step_module):
        attr = getattr(step_module, attr_name)
        if (
            isinstance(attr, type)
            and hasattr(attr, "run")
            and attr_name.endswith("Step")
            and attr_name != "BaseStep"
        ):
            return attr
    raise ImportError(f"No Step class found in {module_path}.step")


def run_pipeline(config_path: Path) -> None:
    """Execute the full pipeline from a config file."""
    pipeline_cfg = load_pipeline_config(config_path)
    data_root = pipeline_cfg.data_root
    results: dict[str, BaseModel] = {}

    enabled_steps = [s for s in pipeline_cfg.steps if s.enabled]
    logger.info(f"Pipeline '{pipeline_cfg.project_name}' with {len(enabled_steps)} steps")

    for entry in enabled_steps:
        logger.info(f"--- Step: {entry.name} ---")

        step_cls = import_step_class(entry.module)
        step_config = load_step_config(Path(entry.config_file), step_cls.config_type)
        step_instance = step_cls(config=step_config, data_root=data_root)

        # Build input from previous step outputs or defaults
        input_data = {}
        if entry.depends_on:
            for dep in entry.depends_on:
                if dep in results:
                    input_data.update(results[dep].model_dump())

        step_input = step_cls.input_type(**input_data) if input_data else step_cls.input_type()
        output = step_instance.execute(step_input)
        results[entry.name] = output

    logger.info("Pipeline complete.")
