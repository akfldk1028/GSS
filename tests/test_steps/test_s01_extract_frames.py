"""Tests for S01: Extract Frames step."""

from pathlib import Path

import numpy as np
import pytest

from gss.steps.s01_extract_frames.config import ExtractFramesConfig
from gss.steps.s01_extract_frames.contracts import ExtractFramesInput, ExtractFramesOutput
from gss.steps.s01_extract_frames.step import ExtractFramesStep


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    """Create a small test video."""
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (160, 120))
    for i in range(90):  # 3 seconds at 30fps
        frame = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestExtractFramesContracts:
    def test_input_validation(self):
        inp = ExtractFramesInput(video_path=Path("/tmp/test.mp4"))
        assert inp.video_path == Path("/tmp/test.mp4")

    def test_output_schema(self):
        schema = ExtractFramesOutput.model_json_schema()
        assert "frames_dir" in schema["properties"]
        assert "frame_count" in schema["properties"]

    def test_config_defaults(self):
        cfg = ExtractFramesConfig()
        assert cfg.target_fps == 2.0
        assert cfg.max_frames == 2000
        assert cfg.blur_threshold == 100.0


class TestExtractFramesStep:
    def test_validate_missing_video(self, data_root: Path):
        cfg = ExtractFramesConfig()
        step = ExtractFramesStep(config=cfg, data_root=data_root)
        inp = ExtractFramesInput(video_path=Path("/nonexistent/video.mp4"))
        assert step.validate_inputs(inp) is False

    def test_extract_frames(self, test_video: Path, data_root: Path):
        cfg = ExtractFramesConfig(target_fps=10.0, blur_threshold=0.0)  # low threshold to get frames
        step = ExtractFramesStep(config=cfg, data_root=data_root)
        inp = ExtractFramesInput(video_path=test_video)

        output = step.execute(inp)
        assert output.frame_count > 0
        assert output.frames_dir.exists()
        assert len(output.frame_list) == output.frame_count

    def test_max_frames_limit(self, test_video: Path, data_root: Path):
        cfg = ExtractFramesConfig(target_fps=30.0, blur_threshold=0.0, max_frames=3)
        step = ExtractFramesStep(config=cfg, data_root=data_root)
        inp = ExtractFramesInput(video_path=test_video)

        output = step.execute(inp)
        assert output.frame_count <= 3
