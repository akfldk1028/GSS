"""Threaded camera capture with frame buffer.

Runs camera in a background thread, accumulates sharp+diverse frames.
Call flush() to retrieve and clear the buffer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureConfig

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray
    index: int
    timestamp: float
    blur_score: float
    motion_score: float


class FrameBuffer:
    """Threaded camera capture with quality filtering."""

    def __init__(self, config: CaptureConfig, output_dir: Path) -> None:
        self._config = config
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: list[Frame] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._prev_gray: np.ndarray | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Capture started (device={self._config.device})")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Capture stopped")

    def flush(self) -> list[Frame]:
        """Return accumulated frames and clear buffer."""
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
        return frames

    def save_frames(self, frames: list[Frame], target_dir: Path) -> list[Path]:
        """Save frame images to disk. Returns list of saved paths."""
        target_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for frame in frames:
            path = target_dir / f"frame_{frame.index:05d}.png"
            cv2.imwrite(str(path), frame.image)
            paths.append(path)
        return paths

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._config.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

        if not cap.isOpened():
            logger.error(f"Cannot open camera device {self._config.device}")
            self._running = False
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera: {w}x{h}")

        while self._running:
            ret, image = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = _blur_score(gray)
            motion = _motion_score(self._prev_gray, gray)
            self._prev_gray = gray

            if blur < self._config.blur_threshold:
                continue
            if motion < self._config.motion_threshold:
                continue

            frame = Frame(
                image=image,
                index=self._frame_count,
                timestamp=time.time(),
                blur_score=blur,
                motion_score=motion,
            )
            self._frame_count += 1

            with self._lock:
                if len(self._buffer) < self._config.max_buffer_size:
                    self._buffer.append(frame)

            time.sleep(0.05)  # ~20fps check rate

        cap.release()


class KinectCapture:
    """Azure Kinect depth + RGB capture → pointcloud PLY + RGB frames."""

    def __init__(self, config: CaptureConfig, output_dir: Path) -> None:
        self._config = config
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0

    def capture_once(self, frames_dir: Path) -> tuple[Path, list[Path]]:
        """Single capture: depth pointcloud + RGB frames.

        Returns (pointcloud_ply_path, list_of_rgb_frame_paths).
        """
        import pyk4a
        from pyk4a import PyK4A, Config

        k4a = PyK4A(Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            camera_fps=pyk4a.FPS.FPS_15,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,  # 120°x120° FOV, 512x512, 0.25-2.88m
            synchronized_images_only=True,
        ))
        k4a.start()

        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []

        try:
            # Capture multiple frames for RGB diversity
            for i in range(min(self._config.max_buffer_size, 20)):
                cap = k4a.get_capture()
                if cap.color is not None:
                    rgb = cap.color[:, :, :3]  # BGRA → BGR
                    path = frames_dir / f"frame_{self._frame_count:05d}.png"
                    cv2.imwrite(str(path), rgb)
                    frame_paths.append(path)
                    self._frame_count += 1
                time.sleep(0.1)

            # Last capture for depth pointcloud
            cap = k4a.get_capture()
            ply_path = self._depth_to_ply(cap, self._output_dir / "kinect_pointcloud.ply")
        finally:
            k4a.stop()

        logger.info(f"Kinect: {len(frame_paths)} RGB frames + pointcloud {ply_path}")
        return ply_path, frame_paths

    def _depth_to_ply(self, capture, ply_path: Path) -> Path:
        """Convert Kinect depth capture to PLY pointcloud."""
        import open3d as o3d

        points = capture.depth_point_cloud.reshape(-1, 3)
        mask = points[:, 2] > 0
        points = points[mask].astype(np.float32) / 1000.0  # mm → meters

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        try:
            tc = capture.transformed_color
            if tc is not None:
                colors = tc[:, :, (2, 1, 0)].reshape(-1, 3) / 255.0  # BGRA → RGB
                colors = colors[mask]
                pcd.colors = o3d.utility.Vector3dVector(colors)
        except RuntimeError:
            logger.warning("transformed_color failed, PLY will have no color")

        pcd = pcd.voxel_down_sample(self._config.depth_voxel_size)

        o3d.io.write_point_cloud(str(ply_path), pcd)
        logger.info(f"Depth pointcloud: {len(pcd.points)} points → {ply_path}")
        return ply_path


def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _motion_score(prev: np.ndarray | None, curr: np.ndarray) -> float:
    if prev is None:
        return 999.0
    return float(cv2.absdiff(prev, curr).mean())
