"""Step 01: Extract frames from video with blur filtering."""

from __future__ import annotations

import logging
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import ExtractFramesConfig
from .contracts import ExtractFramesInput, ExtractFramesOutput

logger = logging.getLogger(__name__)


class ExtractFramesStep(BaseStep[ExtractFramesInput, ExtractFramesOutput, ExtractFramesConfig]):
    name: ClassVar[str] = "extract_frames"
    input_type: ClassVar = ExtractFramesInput
    output_type: ClassVar = ExtractFramesOutput
    config_type: ClassVar = ExtractFramesConfig

    def validate_inputs(self, inputs: ExtractFramesInput) -> bool:
        if not inputs.video_path.exists():
            logger.error(f"Video not found: {inputs.video_path}")
            return False
        return True

    def run(self, inputs: ExtractFramesInput) -> ExtractFramesOutput:
        import cv2

        output_dir = self.data_root / "interim" / "s01_frames"
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(inputs.video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(src_fps / self.config.target_fps))

        extracted = []
        frame_idx = 0
        while cap.isOpened() and len(extracted) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score >= self.config.blur_threshold:
                    if self.config.resize_width:
                        h, w = frame.shape[:2]
                        new_h = int(h * self.config.resize_width / w)
                        frame = cv2.resize(frame, (self.config.resize_width, new_h))
                    fname = f"frame_{len(extracted):05d}.{self.config.output_format}"
                    cv2.imwrite(str(output_dir / fname), frame)
                    extracted.append(fname)
            frame_idx += 1
        cap.release()

        logger.info(f"Extracted {len(extracted)} frames from {total_frames} total (step={step})")
        return ExtractFramesOutput(
            frames_dir=output_dir,
            frame_count=len(extracted),
            fps_used=src_fps / step,
            frame_list=extracted,
        )
