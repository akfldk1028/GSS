"""Azure Kinect → GPS-SLAM 데이터 브릿지.

Kinect에서 RGBD 프레임을 캡처하여 GPS-SLAM이 읽을 수 있는 형식으로 저장.
별도 터미널에서 실행하면 계속 프레임을 쌓고,
GPS-SLAM은 그 폴더를 입력으로 받아 실행.

Usage:
    conda run -n gss python scripts/kinect_to_gpsslam.py --output data/kinect_live --frames 200
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A


def get_kinect_intrinsics(k4a: PyK4A) -> dict:
    """Get color camera intrinsics from Kinect calibration."""
    calib = k4a.calibration
    color = calib.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    fx = float(color[0, 0])
    fy = float(color[1, 1])
    cx = float(color[0, 2])
    cy = float(color[1, 2])
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/kinect_live")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=5, choices=[5, 15, 30])
    args = parser.parse_args()

    out = Path(args.output)
    cam_dir = out / "camera"
    depth_dir = out / "depth"
    cam_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    fps_map = {5: pyk4a.FPS.FPS_5, 15: pyk4a.FPS.FPS_15, 30: pyk4a.FPS.FPS_30}

    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        camera_fps=fps_map[args.fps],
        depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    time.sleep(1)

    # Warmup
    for _ in range(5):
        try:
            k4a.get_capture(timeout=2000)
        except Exception:
            time.sleep(0.3)

    # Get intrinsics
    intrinsics = get_kinect_intrinsics(k4a)
    print(f"Intrinsics: fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
          f"cx={intrinsics['cx']:.1f} cy={intrinsics['cy']:.1f}")

    # Identity pose (GPS-SLAM will do its own tracking)
    identity = np.eye(4)

    count = 0
    try:
        while count < args.frames:
            try:
                cap = k4a.get_capture(timeout=1000)
            except Exception:
                continue

            if cap.color is None or cap.depth is None:
                continue

            # RGB: BGR → save as jpg
            rgb = cap.color[:, :, :3]  # BGRA → BGR
            rgb_path = cam_dir / f"frame{count:06d}.jpg"
            cv2.imwrite(str(rgb_path), rgb)

            # Depth: transformed to color space, uint16 mm
            depth_in_color = cap.transformed_depth  # (720, 1280) uint16 mm
            if depth_in_color is None:
                # Fallback: use raw depth (different resolution)
                continue
            depth_path = depth_dir / f"depth{count:06d}.png"
            cv2.imwrite(str(depth_path), depth_in_color)

            # Pose: identity (SLAM will estimate)
            pose_path = cam_dir / f"pose{count:06d}.txt"
            np.savetxt(str(pose_path), identity, fmt="%.6f")

            count += 1
            if count % 10 == 0:
                print(f"[{count}/{args.frames}] captured", flush=True)

    except KeyboardInterrupt:
        print(f"\nStopped at {count} frames")
    finally:
        k4a.stop()

    # Write config for GPS-SLAM
    h, w = 720, 1280
    config_text = f"""workspace_dir: output/kinect_live
dev_id: 0
work_mode: train

READER:
  input_dir: {out.resolve()}
  image_path: camera
  pose_path: camera
  depth_path: depth
  pcd_name: ""

  depth_scale: 1000.0
  scene_scale: 1.0
  start_frame: 0
  end_frame: {count - 1}
  frame_step: 1
  intrinsics: [{intrinsics['fx']:.1f}, {intrinsics['fy']:.1f}, {intrinsics['cx']:.1f}, {intrinsics['cy']:.1f}]
  image_shape: [{w}, {h}]
  downscale_factor: 1
  test_split_interval: -1

PIPE:
  train_mode: ges
  max_iterations: 10000
  enable_densify: false
  eval_after_train: true
  save_after_train: true
  selected_cam_idx: -1
  model_path: "/gs_model"
  log_path: "/log"
  eval_path: "/val"
  log_iter: 1000
  weight_configs:
    ssim_weight: 0.0
    depth_weight: 0.0
  vis_configs:
    color_error_max: 0.1
    depth_error_max: 0.1
    depth_vis_max: 3
    depth_vis_min: 0
    alpha_vis_max: 5
  log_slam_state: false
  new_gs_sample_ratio: 0.25
  color_error_thres: 0.05
  localframe_cam_window_length: 2
  localframe_cam_window_interval: 5
  local_opt_iters: 20
  local_opt_interval: 10
  keyframe_theta_thres: 30
  keyframe_trans_thres: 0.3
  keyframe_select_max: 7
  keyframe_sample_configs:
    weight_intervel: 0.01
    loss_thres: 0.02
    opt_thres: 50
    sample_method: random
  remove_configs:
    large_scale_thres: 0.1
    small_scale_thres: 0.003
    low_opac_thres: 0.005

  TSDF:
    voxel_size: 0.005
    trunc_dist: 0.02
    viewFrustum_min: 0.25
    viewFrustum_max: 2.88
    load_images: true
    use_gt_pose: false
    saved_mesh: "tsdf_mesh.ply"
    saved_engine: "tsdf_engine/"
    saved_images: raycasted

MODEL:
  render_method: ges
  max_gs_radii: 100
  delta_depth: 0.1
  sh_degree: 3
  sh_degree_interval: 0
  max_init_scale: 0.01
  min_init_scale: -1
  default_opacities: 0.5
  means_lr: 0.00016
  means_lr_final: 0.0000016
  scales_lr: 0.005
  quats_lr: 0.001
  featuresDc_lr: 0.0025
  featuresRest_lr: 0.0005
  opacities_lr: 5e-2
  exposure_lr: 0.003
  use_exposure: false
  densify_start_iter: 500
  densify_end_iter: 6000
  densify_interval: 100
  densify_grad_thres: 0.0002
  densify_large_thres: 0.01
  split_screen_size: 0.05
  reset_opacity_interval: 3000
  prune_opacity_thres: 0.005
"""
    config_path = out / "gpsslam_config.yaml"
    config_path.write_text(config_text)
    print(f"\nDone! {count} frames saved to {out}")
    print(f"Config: {config_path}")
    print(f"\nRun GPS-SLAM:")
    print(f"  cd GSS/clone/GPS-SLAM")
    print(f"  ./build/slam_trainer {config_path.resolve()}")


if __name__ == "__main__":
    main()
