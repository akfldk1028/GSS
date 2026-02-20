"""Base class for all pipeline steps.

Every step declares typed Input, Output, Config via Pydantic models.
This enables AI agents to understand each step independently,
the pipeline runner to validate step connections,
and n8n-style JSON Schema introspection.
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, ClassVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)
ConfigT = TypeVar("ConfigT", bound=BaseModel)

logger = logging.getLogger(__name__)


class BaseStep(ABC, Generic[InputT, OutputT, ConfigT]):
    """Abstract base for pipeline steps.

    Subclasses must:
    1. Define concrete Pydantic models for InputT, OutputT, ConfigT
    2. Set class variables: input_type, output_type, config_type
    3. Implement run() and validate_inputs()

    Example:
        class ExtractFramesStep(BaseStep[FramesInput, FramesOutput, FramesConfig]):
            input_type = FramesInput
            output_type = FramesOutput
            config_type = FramesConfig

            def run(self, inputs: FramesInput) -> FramesOutput: ...
            def validate_inputs(self, inputs: FramesInput) -> bool: ...
    """

    name: ClassVar[str] = ""
    input_type: ClassVar[type[BaseModel]]
    output_type: ClassVar[type[BaseModel]]
    config_type: ClassVar[type[BaseModel]]

    def __init__(self, config: ConfigT, data_root: Path):
        self.config = config
        self.data_root = Path(data_root)

    @abstractmethod
    def run(self, inputs: InputT) -> OutputT:
        """Execute this pipeline step. Returns output model."""
        ...

    @abstractmethod
    def validate_inputs(self, inputs: InputT) -> bool:
        """Check that all required input artifacts exist and are valid."""
        ...

    def execute(self, inputs: InputT) -> OutputT:
        """Run with logging, timing, and validation."""
        step_name = self.name or self.__class__.__name__
        logger.info(f"[{step_name}] Validating inputs...")

        if not self.validate_inputs(inputs):
            raise ValueError(f"[{step_name}] Input validation failed")

        logger.info(f"[{step_name}] Starting...")
        t0 = time.time()
        result = self.run(inputs)
        elapsed = time.time() - t0
        logger.info(f"[{step_name}] Done in {elapsed:.1f}s")
        return result

    @classmethod
    def get_input_schema(cls) -> dict:
        """Return JSON schema for inputs (n8n-style introspection)."""
        return cls.input_type.model_json_schema()

    @classmethod
    def get_output_schema(cls) -> dict:
        """Return JSON schema for outputs."""
        return cls.output_type.model_json_schema()

    @classmethod
    def get_config_schema(cls) -> dict:
        """Return JSON schema for config."""
        return cls.config_type.model_json_schema()
