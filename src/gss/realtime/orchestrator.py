"""Realtime orchestrator — capture → 3DGRUT train+export → Isaac Sim.

Usage:
    python -m gss.realtime.orchestrator
    python -m gss.realtime.orchestrator --config configs/realtime.yaml
    python -m gss.realtime.orchestrator --cycle-interval 20 --device 0
"""

from __future__ import annotations

import json
import logging
import shutil
import signal
import threading
import time
from pathlib import Path

from .capture import FrameBuffer, KinectCapture
from .config import RealtimeConfig
from .exporter import USDZExporter
from .trainer import IncrementalTrainer

logger = logging.getLogger(__name__)


class RealtimeOrchestrator:
    """Main loop: camera → 3DGRUT → USDZ → Isaac Sim."""

    def __init__(self, config: RealtimeConfig) -> None:
        self._config = config
        self._data_root = Path(config.data_root)
        self._data_root.mkdir(parents=True, exist_ok=True)

        self._use_depth = config.capture.use_depth
        if self._use_depth:
            self._kinect = KinectCapture(config.capture, self._data_root)
        self._capture = FrameBuffer(config.capture, self._data_root / "frames")
        self._trainer = IncrementalTrainer(config.trainer, self._data_root)
        self._exporter = USDZExporter(config.exporter)

        self._running = False
        self._cycle_count = 0

    def run(self) -> None:
        """Run continuous capture-train-export loop. Ctrl+C to stop."""
        self._running = True
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: self._stop())

        logger.info(
            f"Realtime pipeline started "
            f"(cycle={self._config.cycle_interval_sec}s, "
            f"min_frames={self._config.min_frames_per_cycle})"
        )
        self._capture.start()

        try:
            while self._running:
                self._stop_event.wait(timeout=self._config.cycle_interval_sec)
                if not self._running:
                    break
                self._run_cycle()
        finally:
            self._capture.stop()
            logger.info(f"Realtime pipeline stopped after {self._cycle_count} cycles")

    def run_once(self, capture_sec: float = 10.0) -> Path | None:
        """Single capture-train-export cycle.

        Starts capture, waits capture_sec seconds, then runs one cycle.
        Returns USDZ path or None.
        """
        if self._use_depth:
            return self._run_depth_cycle()

        logger.info(f"Capturing for {capture_sec}s...")
        self._capture.start()
        time.sleep(capture_sec)
        result = self._run_cycle()
        self._capture.stop()
        return result

    def _run_cycle(self) -> Path | None:
        """One iteration: flush frames → 3DGRUT train+export → notify Isaac."""
        frames = self._capture.flush()
        if len(frames) < self._config.min_frames_per_cycle:
            logger.info(
                f"Cycle {self._cycle_count}: "
                f"only {len(frames)} frames (need {self._config.min_frames_per_cycle}), skipping"
            )
            return None

        logger.info(f"=== Cycle {self._cycle_count}: {len(frames)} frames ===")
        t0 = time.time()

        # Save frames to disk
        cycle_dir = self._data_root / f"cycle_{self._cycle_count:04d}"
        frame_paths = self._capture.save_frames(frames, cycle_dir / "frames")

        # 3DGRUT: COLMAP → train → USDZ (all in one)
        usdz_path = self._trainer.train(frame_paths, cycle_dir)

        # Generate manifest for Isaac Sim
        manifest_path = cycle_dir / "scene_manifest_nurec.json"
        self._exporter._write_manifest(usdz_path, usdz_path, manifest_path)

        # Symlink latest
        latest_dir = self._data_root / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        for name, src in [("scene.nurec.usdz", usdz_path), ("scene_manifest_nurec.json", manifest_path)]:
            dst = latest_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        # Copy to Isaac Sim target if configured
        if self._config.notify_isaac and self._config.isaac_usdz_path:
            target = Path(self._config.isaac_usdz_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(usdz_path, target)
            logger.info(f"USDZ copied to Isaac target: {target}")
            self._write_reload_signal(target.parent)

        elapsed = time.time() - t0
        self._cycle_count += 1
        logger.info(f"Cycle done in {elapsed:.1f}s: {usdz_path}")

        return usdz_path

    def _run_depth_cycle(self) -> Path | None:
        """Kinect depth cycle: capture depth+RGB → 3DGRUT with pointcloud init."""
        logger.info(f"=== Depth Cycle {self._cycle_count} ===")
        t0 = time.time()

        cycle_dir = self._data_root / f"cycle_{self._cycle_count:04d}"
        frames_dir = cycle_dir / "frames"

        ply_path, frame_paths = self._kinect.capture_once(frames_dir)

        if len(frame_paths) < self._config.min_frames_per_cycle:
            logger.info(f"Only {len(frame_paths)} frames, skipping")
            return None

        usdz_path = self._trainer.train_depth_only(ply_path, frame_paths, cycle_dir)

        manifest_path = cycle_dir / "scene_manifest_nurec.json"
        self._exporter._write_manifest(usdz_path, usdz_path, manifest_path)

        latest_dir = self._data_root / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        for name, src in [("scene.nurec.usdz", usdz_path), ("scene_manifest_nurec.json", manifest_path)]:
            dst = latest_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        if self._config.notify_isaac and self._config.isaac_usdz_path:
            target = Path(self._config.isaac_usdz_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(usdz_path, target)
            self._write_reload_signal(target.parent)

        elapsed = time.time() - t0
        self._cycle_count += 1
        logger.info(f"Depth cycle done in {elapsed:.1f}s: {usdz_path}")
        return usdz_path

    def _write_reload_signal(self, target_dir: Path) -> None:
        """Write a signal file for Isaac Sim to detect and reload."""
        signal_file = target_dir / ".reload_trigger"
        signal_file.write_text(
            json.dumps({"cycle": self._cycle_count, "timestamp": time.time()})
        )

    def _stop(self) -> None:
        logger.info("Stopping...")
        self._running = False
        if hasattr(self, "_stop_event"):
            self._stop_event.set()


def main() -> None:
    """CLI entry point."""
    import argparse

    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="GSS Realtime Pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--cycle-interval", type=float, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--once", action="store_true", help="Run single cycle then exit")
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        config = RealtimeConfig(**raw)
    else:
        config = RealtimeConfig()

    # CLI overrides
    if args.cycle_interval is not None:
        config.cycle_interval_sec = args.cycle_interval
    if args.device is not None:
        config.capture.device = args.device
    if args.min_frames is not None:
        config.min_frames_per_cycle = args.min_frames
    if args.data_root is not None:
        config.data_root = Path(args.data_root)

    orchestrator = RealtimeOrchestrator(config)

    if args.once:
        orchestrator.run_once()
    else:
        orchestrator.run()


if __name__ == "__main__":
    main()
