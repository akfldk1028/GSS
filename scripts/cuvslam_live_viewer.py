"""Kinect → PyCuVSLAM → Open3D VBG → viser 실시간 3D 재구성 + BIM 연계.

아키텍처 (2026-04-19 확정):
    Azure Kinect (pyk4a)
        → PyCuVSLAM Tracker (RGBD, feature-based, 회전 robust)
        → Open3D VoxelBlockGrid (외부 포즈 integrate, ICP 없음)
        → viser (실시간 포인트클라우드)
        → [Save PLY] → data/raw/realtime/*.ply → s00_import_ply → BIM

대체: GSS/scripts/kinect_live_viewer.py (Open3D t.slam.Model, KinectFusion 2011 algorithm ceiling).
참고 메모리: memory/gss/realtime-slam-architecture.md
"""

import time
from pathlib import Path

import cuvslam as vslam
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pyk4a
import viser
from pyk4a import Config, PyK4A
from scipy.spatial.transform import Rotation


# ── 파라미터 ─────────────────────────────────────────
DEPTH_SCALE = 1000.0   # Kinect depth (mm) → meters (Open3D VBG는 float)
DEPTH_MIN = 0.3
DEPTH_MAX = 3.0
VOXEL_SIZE = 0.006     # Open3D 공식 default (6mm)
BLOCK_RES = 16
BLOCK_COUNT = 40000
SURFACE_WEIGHT_THR = 3.0
EST_POINT_COUNT = 3_000_000
EXTRACT_EVERY = 10
PLY_SAVE_DIR = Path("/home/cgxr/Documents/DK/DTR/GSS/data/raw/realtime")


def pose_to_matrix(pose: vslam.Pose) -> np.ndarray:
    """cuvslam.Pose(rotation=quat xyzw, translation=[x,y,z]) → 4x4 matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(list(pose.rotation)).as_matrix()
    T[:3, 3] = list(pose.translation)
    return T


def make_vbg(device):
    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=VOXEL_SIZE,
        block_resolution=BLOCK_RES,
        block_count=BLOCK_COUNT,
        device=device,
    )


def main():
    PLY_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── viser 서버 + GUI ────────────────────────────
    server = viser.ViserServer(port=8080)
    server.scene.world_axes.visible = True
    server.scene.add_grid("/grid", width=10.0, height=10.0, plane="xz")
    server.scene.set_up_direction((0.0, 1.0, 0.0))

    gui_point_size = server.gui.add_slider("Point Size", 0.001, 0.02, 0.001, 0.005)
    gui_save = server.gui.add_button("Save PLY → BIM 파이프라인")
    gui_reset = server.gui.add_button("Reset TSDF")
    gui_status = server.gui.add_text("Status", initial_value="Starting...")

    save_flag = [False]
    reset_flag = [False]

    @gui_save.on_click
    def _(_): save_flag[0] = True

    @gui_reset.on_click
    def _(_): reset_flag[0] = True

    @server.on_client_connect
    def _(c):
        c.camera.position = np.array([0.0, 2.0, -3.0])
        c.camera.look_at = np.array([0.0, 0.0, 2.0])
        c.camera.up_direction = np.array([0.0, 1.0, 0.0])

    # ── Kinect ──────────────────────────────────────
    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        camera_fps=pyk4a.FPS.FPS_15,
        depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    time.sleep(1.0)

    # depth intrinsic (512x512 for WFOV_2X2BINNED)
    K_np = np.asarray(
        k4a.calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH),
        dtype=np.float64,
    )
    fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
    cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
    W, H = 512, 512
    print(f"Intrinsic fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}", flush=True)

    # warm up Kinect
    for _ in range(10):
        try:
            k4a.get_capture(timeout=2000)
        except Exception:
            time.sleep(0.3)

    # ── PyCuVSLAM Tracker ───────────────────────────
    cam = vslam.Camera(
        size=(W, H),
        principal=(cx, cy),
        focal=(fx, fy),
        distortion=vslam.Distortion(model=vslam.Distortion.Model.Pinhole, parameters=[]),
    )
    rig = vslam.Rig(cameras=[cam])
    odom_cfg = vslam.core.Odometry.Config(
        odometry_mode=vslam.core.Odometry.OdometryMode.RGBD,
        use_gpu=True,
        async_sba=True,
        rgbd_settings=vslam.core.Odometry.RGBDSettings(
            depth_scale_factor=0.001,   # mm → m
            depth_camera_id=0,          # camera[0] provides depth
            enable_depth_stereo_tracking=False,
        ),
    )
    tracker = vslam.Tracker(rig, odom_config=odom_cfg)
    print("PyCuVSLAM tracker initialized (RGBD mode, GPU)", flush=True)

    # ── Open3D VBG (CUDA) ───────────────────────────
    device = o3c.Device("CUDA:0")
    vbg = make_vbg(device)
    intrinsic_t = o3c.Tensor(K_np, o3c.Dtype.Float64)

    flip = np.diag([-1.0, -1.0, 1.0, 1.0])  # Kinect (OpenCV) → viser Y-up 표시용
    frame_id = 0
    ok = fail = 0
    last_valid_pose = np.eye(4)

    try:
        while True:
            try:
                cap = k4a.get_capture(timeout=1000)
            except Exception:
                continue

            if reset_flag[0]:
                del vbg
                vbg = make_vbg(device)
                frame_id = 0
                ok = fail = 0
                last_valid_pose = np.eye(4)
                reset_flag[0] = False
                gui_status.value = "Reset."
                continue

            if cap.depth is None or cap.color is None:
                continue
            try:
                tc = cap.transformed_color  # BGRA, aligned to depth
            except RuntimeError:
                continue
            if tc is None:
                continue

            rgb = np.ascontiguousarray(tc[:, :, (2, 1, 0)].astype(np.uint8))
            depth_u16 = np.ascontiguousarray(cap.depth.astype(np.uint16))
            t_ns = int(cap.depth_timestamp_usec * 1000)

            # ── PyCuVSLAM 트래킹 ──
            try:
                pose_est, _slam_pose = tracker.track(
                    timestamp=t_ns,
                    images=[rgb],
                    depths=[depth_u16],
                )
            except Exception as e:
                print(f"  [track err f{frame_id}] {type(e).__name__}: {e}", flush=True)
                fail += 1
                frame_id += 1
                continue

            if pose_est.world_from_rig is None:
                # 트래킹 실패 — 이 프레임 integrate 스킵, pose 유지
                fail += 1
                frame_id += 1
                if frame_id % 30 == 0:
                    gui_status.value = f"TRACK LOST (fail={fail})"
                continue

            T_wc = pose_to_matrix(pose_est.world_from_rig.pose)
            last_valid_pose = T_wc
            ok += 1

            # ── Open3D VBG integrate ──
            T_cw = np.linalg.inv(T_wc)  # VBG는 world→camera (extrinsic)
            extrinsic = o3c.Tensor(T_cw, o3c.Dtype.Float64)

            depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_u16)).to(device)
            color_t = o3d.t.geometry.Image(o3c.Tensor(rgb)).to(device)

            try:
                blocks = vbg.compute_unique_block_coordinates(
                    depth_t, intrinsic_t, extrinsic, DEPTH_SCALE, DEPTH_MAX,
                )
                vbg.integrate(
                    blocks, depth_t, color_t, intrinsic_t, intrinsic_t, extrinsic,
                    DEPTH_SCALE, DEPTH_MAX,
                )
            except Exception as e:
                print(f"  [integrate err f{frame_id}] {type(e).__name__}: {e}", flush=True)

            frame_id += 1

            # ── 주기적 추출 + viser 표시 ──
            if frame_id % EXTRACT_EVERY == 0:
                try:
                    pcd_t = vbg.extract_point_cloud(
                        SURFACE_WEIGHT_THR, EST_POINT_COUNT,
                    ).to(o3c.Device("CPU:0"))
                    pcd = pcd_t.to_legacy()
                except Exception:
                    continue

                pcd.transform(flip)  # viser Y-up
                pts = np.asarray(pcd.points, dtype=np.float32)
                cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None

                if len(pts) > 0:
                    n_display = min(len(pts), 500_000)
                    idx = np.random.choice(len(pts), n_display, replace=False) if len(pts) > n_display else np.arange(len(pts))
                    display_pts = pts[idx]
                    display_cols = cols[idx] if cols is not None else np.full((n_display, 3), 200, dtype=np.uint8)
                    server.scene.add_point_cloud(
                        "/realtime_map",
                        points=display_pts,
                        colors=display_cols,
                        point_size=gui_point_size.value,
                        point_shape="rounded",
                    )
                    gui_status.value = f"{len(pts)} pts | ok={ok} fail={fail}"

                # ── Save PLY ──
                if save_flag[0]:
                    ts = int(time.time())
                    save_path = PLY_SAVE_DIR / f"realtime_{ts}.ply"
                    # 원본 좌표계로 저장 (flip 이전)
                    pcd_orig = pcd_t.to_legacy()  # CPU에 있는 원본
                    o3d.io.write_point_cloud(str(save_path), pcd_orig)
                    print(f"SAVED: {save_path} ({len(pcd_orig.points)} pts)", flush=True)
                    gui_status.value = f"SAVED {save_path.name} ({len(pcd_orig.points)} pts)"
                    save_flag[0] = False

                if frame_id % 30 == 0:
                    print(f"[{frame_id}] {len(pts)} pts, ok={ok} fail={fail}", flush=True)

            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()


if __name__ == "__main__":
    main()
