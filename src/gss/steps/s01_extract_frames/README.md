# S01: Extract Frames

Extract frames from video with blur filtering and optional resize.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `video_path` | Path | Input video file (.mp4, .avi, etc.) |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `frames_dir` | Path | `data/interim/s01_frames/` |
| `frame_count` | int | Number of extracted frames |
| `fps_used` | float | Effective extraction FPS |
| `frame_list` | list[str] | Frame filenames |

## Config (`configs/steps/s01_extract_frames.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `target_fps` | 2.0 | Target extraction rate |
| `max_frames` | 2000 | Maximum frames to extract |
| `output_format` | png | Image format (png/jpg) |
| `blur_threshold` | 100.0 | Laplacian variance threshold |
| `resize_width` | null | Resize width (null = original) |

## Dependencies
- `opencv-python >= 4.8`
- `numpy`
