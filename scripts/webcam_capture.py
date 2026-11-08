#!/usr/bin/env python3
"""webcam_capture.py — Real-time webcam capture for GSS pipeline.

Captures frames from webcam with blur filtering, saves to a directory
that can be directly fed into the GSS pipeline (skipping s01).

Usage:
    # Capture frames (press 'q' to stop, 's' to force-save current frame)
    python scripts/webcam_capture.py --output data/raw/webcam_session

    # Then run GSS from s02 (COLMAP) with captured frames
    python cli.py run-step colmap --frames-dir data/raw/webcam_session/frames

    # Or run the full pipeline with webcam video
    python scripts/webcam_capture.py --output data/raw/webcam_session --save-video
    python cli.py run --video data/raw/webcam_session/capture.mp4

Controls:
    q     — Stop capture
    s     — Force-save current frame (even if blurry)
    SPACE — Pause/resume auto-capture
    +/-   — Adjust capture interval

Requires: opencv-python (cv2)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def compute_blur_score(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_motion_score(prev_gray: np.ndarray | None, curr_gray: np.ndarray) -> float:
    """Frame difference — higher = more motion (good for diverse viewpoints)."""
    if prev_gray is None:
        return 999.0
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean())


def main():
    parser = argparse.ArgumentParser(description="Webcam capture for GSS pipeline")
    parser.add_argument("--output", type=str, default="data/raw/webcam_session",
                        help="Output directory")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Capture height (default: 720)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between auto-captures (default: 0.5)")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Minimum Laplacian variance to keep frame")
    parser.add_argument("--motion-threshold", type=float, default=5.0,
                        help="Minimum motion to trigger capture")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Maximum frames to capture")
    parser.add_argument("--save-video", action="store_true",
                        help="Also save raw video as capture.mp4")
    parser.add_argument("--resize-output", type=int, default=None,
                        help="Resize saved frames to this width")
    parser.add_argument("--headless", action="store_true",
                        help="No GUI window (for SSH/headless systems)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        logger.error(f"Cannot open camera device {args.device}")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

    # Video writer (optional)
    video_writer = None
    if args.save_video:
        video_path = output_dir / "capture.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(video_path), fourcc, actual_fps, (actual_w, actual_h)
        )
        logger.info(f"Recording video to {video_path}")

    # State
    saved_frames: list[dict] = []
    prev_gray = None
    paused = False
    interval = args.interval
    last_capture_time = 0.0

    logger.info(
        f"Capture started. Controls: q=quit, s=force-save, SPACE=pause, +/-=interval"
    )
    logger.info(
        f"blur_threshold={args.blur_threshold}, motion_threshold={args.motion_threshold}, "
        f"interval={interval}s, max_frames={args.max_frames}"
    )

    while cap.isOpened() and len(saved_frames) < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = compute_blur_score(frame)
        motion = compute_motion_score(prev_gray, gray)

        # Record video
        if video_writer is not None:
            video_writer.write(frame)

        # Auto-capture logic
        should_save = False
        force_save = False

        if not args.headless:
            # GUI mode — display info overlay
            display = frame.copy()
            status = "PAUSED" if paused else "CAPTURING"
            color = (0, 0, 255) if paused else (0, 255, 0)
            info_lines = [
                f"{status} | Frames: {len(saved_frames)}/{args.max_frames}",
                f"Blur: {blur:.0f} (min {args.blur_threshold:.0f}) | Motion: {motion:.1f}",
                f"Interval: {interval:.1f}s | q=quit s=save SPACE=pause",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(display, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            blur_ratio = min(blur / max(args.blur_threshold * 2, 1), 1.0)
            bar_w = int(blur_ratio * 200)
            bar_color = (0, 255, 0) if blur >= args.blur_threshold else (0, 0, 255)
            cv2.rectangle(display, (10, actual_h - 30), (10 + bar_w, actual_h - 10), bar_color, -1)
            cv2.rectangle(display, (10, actual_h - 30), (210, actual_h - 10), (255, 255, 255), 1)

            cv2.imshow("GSS Webcam Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                force_save = True
                should_save = True
            elif key == ord(" "):
                paused = not paused
                logger.info(f"{'Paused' if paused else 'Resumed'} capture")
            elif key == ord("+") or key == ord("="):
                interval = max(0.1, interval - 0.1)
                logger.info(f"Interval: {interval:.1f}s")
            elif key == ord("-"):
                interval = min(5.0, interval + 0.1)
                logger.info(f"Interval: {interval:.1f}s")

        if not paused and not force_save and (now - last_capture_time) >= interval:
            if blur >= args.blur_threshold and motion >= args.motion_threshold:
                should_save = True

        if should_save:
            # Optionally resize
            save_frame = frame
            if args.resize_output:
                h, w = frame.shape[:2]
                new_h = int(h * args.resize_output / w)
                save_frame = cv2.resize(frame, (args.resize_output, new_h))

            fname = f"frame_{len(saved_frames):05d}.png"
            cv2.imwrite(str(frames_dir / fname), save_frame)
            saved_frames.append({
                "filename": fname,
                "blur_score": round(blur, 1),
                "motion_score": round(motion, 1),
                "timestamp": round(now - last_capture_time, 3) if last_capture_time > 0 else 0,
                "forced": force_save,
            })
            last_capture_time = now

            if len(saved_frames) % 10 == 0:
                logger.info(f"Captured {len(saved_frames)} frames")

        prev_gray = gray

    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    # Save metadata
    metadata = {
        "capture_date": datetime.now().isoformat(),
        "device": args.device,
        "resolution": [actual_w, actual_h],
        "fps": actual_fps,
        "num_frames": len(saved_frames),
        "blur_threshold": args.blur_threshold,
        "motion_threshold": args.motion_threshold,
        "frames": saved_frames,
    }
    meta_path = output_dir / "capture_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done: {len(saved_frames)} frames saved to {frames_dir}")
    logger.info(f"Metadata: {meta_path}")
    logger.info(f"Next: python cli.py run --video {output_dir / 'capture.mp4'}")
    logger.info(f"  or: python cli.py run-step colmap (with frames_dir={frames_dir})")


if __name__ == "__main__":
    main()
