"""Kinect → GPS-SLAM → viser 실시간 루프.

3개 스레드로 연속 동작:
  1. Kinect 캡처: RGBD 프레임을 계속 디스크에 저장
  2. GPS-SLAM 실행: N프레임 쌓일 때마다 GPS-SLAM 재실행
  3. viser 뷰어: GPS-SLAM 출력 변경 시 자동 reload

Usage:
    conda run -n gss python scripts/gpsslam_realtime_loop.py
    conda run -n gss python scripts/gpsslam_realtime_loop.py --batch 30 --fps 5
    conda run -n gss python scripts/gpsslam_realtime_loop.py --no-kinect  # 기존 데이터로 SLAM+뷰어만
"""

import argparse
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import viser

# GPS-SLAM paths
GPSSLAM_DIR = Path(__file__).resolve().parent.parent / "clone" / "GPS-SLAM"
SLAM_BIN = GPSSLAM_DIR / "build" / "slam_trainer"
GPSSLAM_OUTPUT = GPSSLAM_DIR / "output" / "kinect_live"

# Shared state
_shutdown = threading.Event()
_slam_running = threading.Lock()
_frame_count = 0
_frame_lock = threading.Lock()
_slam_cycle = 0  # incremented after each SLAM run
_viewer_update = threading.Event()


def get_kinect_intrinsics(k4a) -> dict:
    """Get color camera intrinsics from Kinect calibration."""
    import pyk4a
    calib = k4a.calibration
    color = calib.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    return {
        "fx": float(color[0, 0]),
        "fy": float(color[1, 1]),
        "cx": float(color[0, 2]),
        "cy": float(color[1, 2]),
    }


def write_gpsslam_config(data_dir: Path, end_frame: int, intrinsics: dict,
                         start_frame: int = 0, frame_step: int = 1) -> Path:
    """Write/update GPS-SLAM config YAML."""
    config_text = f"""workspace_dir: output/kinect_live
dev_id: 0
work_mode: train

READER:
  input_dir: {data_dir.resolve()}
  image_path: camera
  pose_path: camera
  depth_path: depth
  pcd_name: "none.ply"

  depth_scale: 1000.0
  scene_scale: 1.0
  start_frame: {start_frame}
  end_frame: {end_frame}
  frame_step: {frame_step}
  intrinsics: [{intrinsics['fx']:.1f}, {intrinsics['fy']:.1f}, {intrinsics['cx']:.1f}, {intrinsics['cy']:.1f}]
  image_shape: [1280, 720]
  downscale_factor: 1
  test_split_interval: -1

PIPE:
  train_mode: ges
  max_iterations: 10000
  enable_densify: false
  eval_after_train: false
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
  keyframe_theta_thres: 45
  keyframe_trans_thres: 0.5
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
    config_path = data_dir / "gpsslam_config.yaml"
    config_path.write_text(config_text)
    return config_path


# ---------------------------------------------------------------------------
# Thread 1: Kinect capture
# ---------------------------------------------------------------------------
def kinect_capture_thread(data_dir: Path, fps: int):
    """Continuously capture RGBD frames from Kinect."""
    global _frame_count
    import pyk4a
    from pyk4a import Config, PyK4A

    fps_map = {5: pyk4a.FPS.FPS_5, 15: pyk4a.FPS.FPS_15, 30: pyk4a.FPS.FPS_30}

    cam_dir = data_dir / "camera"
    depth_dir = data_dir / "depth"
    cam_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        camera_fps=fps_map[fps],
        depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    time.sleep(1)

    # Warmup
    for _ in range(5):
        if _shutdown.is_set():
            break
        try:
            k4a.get_capture(timeout=2000)
        except Exception:
            time.sleep(0.3)

    # Get and save intrinsics
    intrinsics = get_kinect_intrinsics(k4a)
    intrinsics_path = data_dir / "intrinsics.npy"
    np.save(str(intrinsics_path), intrinsics)
    print(f"[capture] Intrinsics: fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
          f"cx={intrinsics['cx']:.1f} cy={intrinsics['cy']:.1f}")

    identity = np.eye(4)

    try:
        while not _shutdown.is_set():
            try:
                cap = k4a.get_capture(timeout=1000)
            except Exception:
                continue

            if cap.color is None or cap.depth is None:
                continue

            depth_in_color = cap.transformed_depth
            if depth_in_color is None:
                continue

            with _frame_lock:
                idx = _frame_count
                _frame_count += 1

            # Save RGB
            rgb = cap.color[:, :, :3]
            cv2.imwrite(str(cam_dir / f"frame{idx:06d}.jpg"), rgb)

            # Save depth
            cv2.imwrite(str(depth_dir / f"depth{idx:06d}.png"), depth_in_color)

            # Save identity pose
            np.savetxt(str(cam_dir / f"pose{idx:06d}.txt"), identity, fmt="%.6f")

            if (idx + 1) % 10 == 0:
                print(f"[capture] {idx + 1} frames", flush=True)

    finally:
        k4a.stop()
        print("[capture] stopped")


# ---------------------------------------------------------------------------
# Thread 2: GPS-SLAM runner
# ---------------------------------------------------------------------------
def _accumulate_pointcloud(accumulated: o3d.geometry.PointCloud,
                           new_pcd: o3d.geometry.PointCloud,
                           voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """Merge new point cloud into accumulated using ICP alignment + voxel dedup."""
    if len(accumulated.points) == 0:
        return new_pcd.voxel_down_sample(voxel_size)

    # ICP: align new_pcd to accumulated
    threshold = voxel_size * 5
    reg = o3d.pipelines.registration.registration_icp(
        new_pcd, accumulated, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )

    if reg.fitness < 0.1:
        # ICP failed (no overlap) — just append without alignment
        print(f"[accum] ICP fitness {reg.fitness:.3f} (low), appending raw")
        merged = accumulated + new_pcd
    else:
        print(f"[accum] ICP fitness {reg.fitness:.3f}, inlier_rmse {reg.inlier_rmse:.4f}")
        new_aligned = new_pcd.transform(reg.transformation)
        merged = accumulated + new_aligned

    # Voxel downsample to prevent unbounded growth
    merged = merged.voxel_down_sample(voxel_size)
    return merged


def _accumulate_mesh(accumulated_mesh_path: Path, new_mesh_path: Path,
                     voxel_size: float = 0.005) -> None:
    """Merge new TSDF mesh into accumulated mesh file."""
    if not new_mesh_path.exists():
        return
    new_mesh = o3d.io.read_triangle_mesh(str(new_mesh_path))
    if len(new_mesh.vertices) == 0:
        return

    if accumulated_mesh_path.exists():
        acc_mesh = o3d.io.read_triangle_mesh(str(accumulated_mesh_path))
        if len(acc_mesh.vertices) > 0:
            # Use point-based ICP on mesh vertices to align
            acc_pcd = acc_mesh.sample_points_uniformly(min(50000, len(acc_mesh.vertices)))
            new_pcd = new_mesh.sample_points_uniformly(min(50000, len(new_mesh.vertices)))
            threshold = voxel_size * 10
            reg = o3d.pipelines.registration.registration_icp(
                new_pcd, acc_pcd, threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )
            if reg.fitness > 0.1:
                new_mesh.transform(reg.transformation)
            merged = acc_mesh + new_mesh
            # Simplify if too many vertices (> 3M)
            if len(merged.vertices) > 3_000_000:
                merged = merged.simplify_vertex_clustering(voxel_size * 2)
            o3d.io.write_triangle_mesh(str(accumulated_mesh_path), merged)
            return

    # First cycle or no existing — just copy
    o3d.io.write_triangle_mesh(str(accumulated_mesh_path), new_mesh)


def slam_runner_thread(data_dir: Path, batch_size: int, max_frames: int,
                       frame_step: int, intrinsics_override: dict | None):
    """Run GPS-SLAM whenever enough new frames accumulate.

    Accumulation mode: each cycle's result is ICP-aligned and merged into
    an ever-growing point cloud and mesh, so the reconstruction expands
    as the Kinect moves.
    """
    global _slam_cycle

    stable_dir = GPSSLAM_OUTPUT.parent / "kinect_live_stable"
    stable_dir.mkdir(parents=True, exist_ok=True)

    # Accumulated point cloud (in memory)
    accumulated_pcd = o3d.geometry.PointCloud()

    # Wait for first batch of frames (or intrinsics file from capture thread)
    print(f"[slam] Waiting for {batch_size} frames (max window: {max_frames})...")
    while not _shutdown.is_set():
        with _frame_lock:
            n = _frame_count
        if n >= batch_size:
            break
        time.sleep(1)

    if _shutdown.is_set():
        return

    # Load intrinsics
    if intrinsics_override:
        intrinsics = intrinsics_override
    else:
        intrinsics_path = data_dir / "intrinsics.npy"
        for _ in range(30):
            if intrinsics_path.exists():
                intrinsics = np.load(str(intrinsics_path), allow_pickle=True).item()
                break
            time.sleep(1)
        else:
            print("[slam] ERROR: intrinsics not found after 30s")
            return

    last_processed = 0
    env = {
        "HOME": os.environ["HOME"],
        "PATH": "/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": ":".join([
            "/usr/local/cuda-12.8/lib64",
            str(GPSSLAM_DIR / "ThirdLibs" / "install" / "lib"),
            str(GPSSLAM_DIR / "ThirdLibs" / "libtorch" / "lib"),
            "/usr/lib/x86_64-linux-gnu",
        ]),
    }

    while not _shutdown.is_set():
        with _frame_lock:
            n = _frame_count

        new_frames = n - last_processed
        if new_frames < batch_size:
            time.sleep(1)
            continue

        # Sliding window: use at most max_frames recent frames
        start_frame = max(0, n - max_frames)
        end_frame = n - 1
        window_size = end_frame - start_frame + 1
        config_path = write_gpsslam_config(data_dir, end_frame, intrinsics,
                                           start_frame=start_frame,
                                           frame_step=frame_step)

        print(f"\n[slam] === Cycle {_slam_cycle + 1}: frames {start_frame}-{end_frame} "
              f"({window_size} in window, {n} total captured) ===")

        with _slam_running:
            t0 = time.time()
            proc = subprocess.run(
                [str(SLAM_BIN), str(config_path)],
                cwd=str(GPSSLAM_DIR),
                env=env,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"[slam] FAILED (rc={proc.returncode})")
            for line in proc.stderr.strip().split("\n")[-5:]:
                if "capturesync" not in line:
                    print(f"  stderr: {line}")
            time.sleep(5)
            continue

        # Parse FPS from stdout
        fps_line = ""
        for line in proc.stdout.split("\n"):
            if "FPS" in line or "fps" in line:
                fps_line = line.strip()

        _slam_cycle += 1
        last_processed = n
        print(f"[slam] Done in {elapsed:.1f}s. {fps_line}")

        # --- Accumulate point cloud ---
        gs_src = GPSSLAM_OUTPUT / "gs_model" / "point_cloud" / "point_cloud.ply"
        if gs_src.exists():
            new_pcd = o3d.io.read_point_cloud(str(gs_src))
            if len(new_pcd.points) > 0:
                accumulated_pcd = _accumulate_pointcloud(accumulated_pcd, new_pcd)
                o3d.io.write_point_cloud(str(stable_dir / "point_cloud.ply"),
                                         accumulated_pcd)
                print(f"[accum] Total: {len(accumulated_pcd.points)} points")

        # --- Accumulate mesh ---
        mesh_src = GPSSLAM_OUTPUT / "tsdf_mesh.ply"
        _accumulate_mesh(stable_dir / "tsdf_mesh.ply", mesh_src)

        # Signal viewer to reload
        _viewer_update.set()


# ---------------------------------------------------------------------------
# Thread 3: viser viewer with auto-reload
# ---------------------------------------------------------------------------
def viewer_thread(port: int):
    """viser viewer that reloads GPS-SLAM output when updated."""
    import trimesh as tm

    server = viser.ViserServer(port=port)
    print(f"[viewer] http://localhost:{port}")

    stable_dir = GPSSLAM_OUTPUT.parent / "kinect_live_stable"
    mesh_path = stable_dir / "tsdf_mesh.ply"
    pcd_path = stable_dir / "point_cloud.ply"

    # Add reference grid and axes
    server.scene.add_grid("/grid", width=5, height=5, cell_size=0.5,
                          position=(0, 0, 2))

    # Status label
    cycle_label = server.scene.add_label("/status", text="Waiting for SLAM...")

    loaded_cycle = 0

    while not _shutdown.is_set():
        # Wait for SLAM to signal, or timeout every 3s to check shutdown
        _viewer_update.wait(timeout=3.0)
        if not _viewer_update.is_set():
            continue
        _viewer_update.clear()

        loaded_cycle += 1

        # Reload mesh with vertex colors via trimesh
        if mesh_path.exists():
            try:
                mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
                mesh_o3d.compute_vertex_normals()
                verts = np.asarray(mesh_o3d.vertices)
                faces = np.asarray(mesh_o3d.triangles)

                if mesh_o3d.has_vertex_colors():
                    colors_f = np.asarray(mesh_o3d.vertex_colors)
                    colors_u8 = (colors_f * 255).clip(0, 255).astype(np.uint8)
                    # trimesh wants RGBA per vertex
                    rgba = np.hstack([colors_u8,
                                      np.full((len(colors_u8), 1), 255, dtype=np.uint8)])
                else:
                    rgba = np.full((len(verts), 4), [180, 180, 180, 255], dtype=np.uint8)

                tmesh = tm.Trimesh(vertices=verts, faces=faces,
                                   vertex_colors=rgba, process=False)
                server.scene.add_mesh_trimesh("/mesh", tmesh)
                print(f"[viewer] Mesh: {len(verts)} verts, {len(faces)} faces (vertex colors)")
            except Exception as e:
                print(f"[viewer] Mesh load error: {e}")

        # Reload gaussians as colored point cloud
        if pcd_path.exists():
            try:
                pcd = o3d.io.read_point_cloud(str(pcd_path))
                pts = np.asarray(pcd.points).astype(np.float32)
                if pcd.has_colors():
                    cols = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
                else:
                    # Cyan color for visibility
                    cols = np.full((len(pts), 3), [0, 220, 255], dtype=np.uint8)

                server.scene.add_point_cloud(
                    "/gaussians",
                    points=pts,
                    colors=cols,
                    point_size=0.01,
                    point_shape="rounded",
                )
                print(f"[viewer] Gaussians: {len(pts)} points")
            except Exception as e:
                print(f"[viewer] PCD load error: {e}")

        with _frame_lock:
            n = _frame_count
        cycle_label.text = f"Cycle {loaded_cycle} | {n} frames"
        print(f"[viewer] Updated (cycle {loaded_cycle})")


def main():
    parser = argparse.ArgumentParser(description="Kinect → GPS-SLAM → viser realtime loop")
    parser.add_argument("--data", type=str, default="data/kinect_live",
                        help="Data directory for frames")
    parser.add_argument("--batch", type=int, default=50,
                        help="Run SLAM every N new frames")
    parser.add_argument("--max-frames", type=int, default=600,
                        help="Max frames per SLAM run (sliding window to prevent OOM)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Frame step for SLAM (2=skip every other, reduces motion blur)")
    parser.add_argument("--fps", type=int, default=5, choices=[5, 15, 30],
                        help="Kinect capture FPS")
    parser.add_argument("--port", type=int, default=8080,
                        help="viser viewer port")
    parser.add_argument("--no-kinect", action="store_true",
                        help="Skip Kinect capture, use existing data")
    parser.add_argument("--clean", action="store_true",
                        help="Clean existing frames before starting")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent.parent / args.data

    if args.clean:
        import shutil
        for d in ["camera", "depth"]:
            p = data_dir / d
            if p.exists():
                shutil.rmtree(p)
                print(f"Cleaned {p}")

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "camera").mkdir(exist_ok=True)
    (data_dir / "depth").mkdir(exist_ok=True)

    # Verify SLAM binary
    if not SLAM_BIN.exists():
        print(f"ERROR: GPS-SLAM binary not found: {SLAM_BIN}")
        return

    global _frame_count

    intrinsics_override = None
    if args.no_kinect:
        # Count existing frames
        existing = sorted((data_dir / "camera").glob("frame*.jpg"))
        _frame_count = len(existing)
        print(f"[no-kinect] Found {_frame_count} existing frames")
        if _frame_count == 0:
            print("ERROR: No existing frames found")
            return
        # Use saved intrinsics or default
        intrinsics_path = data_dir / "intrinsics.npy"
        if intrinsics_path.exists():
            intrinsics_override = np.load(str(intrinsics_path), allow_pickle=True).item()
        else:
            intrinsics_override = {"fx": 606.3, "fy": 606.2, "cx": 640.2, "cy": 365.8}
            print(f"[no-kinect] Using default intrinsics: {intrinsics_override}")

    # Signal handler
    def on_signal(sig, frame):
        print("\nShutting down...")
        _shutdown.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    threads = []

    # Start viewer first
    t_viewer = threading.Thread(target=viewer_thread, args=(args.port,), daemon=True)
    t_viewer.start()
    threads.append(t_viewer)

    # Start SLAM runner
    t_slam = threading.Thread(
        target=slam_runner_thread,
        args=(data_dir, args.batch, args.max_frames, args.frame_step, intrinsics_override),
        daemon=True,
    )
    t_slam.start()
    threads.append(t_slam)

    # Start Kinect capture (unless --no-kinect)
    if not args.no_kinect:
        t_capture = threading.Thread(
            target=kinect_capture_thread,
            args=(data_dir, args.fps),
            daemon=True,
        )
        t_capture.start()
        threads.append(t_capture)

    print(f"\n=== Realtime loop started ===")
    print(f"  Data:   {data_dir}")
    print(f"  Batch:  {args.batch} frames")
    print(f"  Viewer: http://localhost:{args.port}")
    print(f"  Ctrl+C to stop\n")

    # Wait for shutdown
    while not _shutdown.is_set():
        time.sleep(0.5)

    # Give threads a moment to clean up
    time.sleep(2)
    print("Done.")


if __name__ == "__main__":
    main()
