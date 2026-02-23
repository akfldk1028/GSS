"""Create MP4 video from Replica room0 frames for pipeline testing."""

from pathlib import Path
import cv2
import sys


def create_video(
    frames_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    max_frames: int = 200,
) -> int:
    """Convert Replica JPG frames to MP4.

    Returns number of frames written.
    """
    frame_files = sorted(frames_dir.glob("frame*.jpg"))
    if not frame_files:
        print(f"ERROR: No frame*.jpg files in {frames_dir}")
        return 0

    frame_files = frame_files[:max_frames]

    # Read first frame to get dimensions
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for f in frame_files:
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)

    writer.release()
    print(f"Created {output_path}: {len(frame_files)} frames, {w}x{h} @ {fps}fps")
    return len(frame_files)


if __name__ == "__main__":
    replica_dir = Path("data/test/replica/Replica/room0/results")
    output = Path("data/raw/replica_room0.mp4")
    max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    create_video(replica_dir, output, max_frames=max_frames)
