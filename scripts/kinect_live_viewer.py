"""⚠️ DEPRECATED 2026-04-19 — Open3D t.slam.Model은 KinectFusion 2011 알고리즘.
회전 drift는 논문 명시 failure mode로 튜닝 불가 (algorithm ceiling).

→ 대체: GSS/scripts/cuvslam_live_viewer.py (PyCuVSLAM + Open3D VoxelBlockGrid)
→ 결정 근거: memory/gss/slam-decision-2026-04-19.md
→ 아키텍처: memory/gss/realtime-slam-architecture.md
→ HANDOFF: HANDOFF/kinect-realtime/HANDOFF.md

이 파일은 비교/참조용으로 보존. 새로 쓰지 말 것.

----
Kinect → viser 실시간 3D 재구성 (KinectFusion frame-to-model).

Open3D `t.pipelines.slam.Model` 사용 — TSDF를 world frame에 유지하고,
매 프레임 raycast된 virtual depth와 ICP하여 frame-to-model 트래킹.
"""

import time
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pyk4a
import viser
from pyk4a import Config, PyK4A


DEPTH_SCALE = 1000.0   # Kinect depth (mm) → meters
DEPTH_MIN = 0.3
DEPTH_MAX = 3.0        # Open3D 공식 default
VOXEL_SIZE = 0.006     # Open3D 공식 default (0.0058) → 6mm
BLOCK_RES = 16
BLOCK_COUNT = 40000
TRUNC_MULT = 8.0
ODO_DIST_THR = 0.15    # 기본 0.07은 회전 시 outlier로 오판되는 depth 너무 많음 → 0.15
ODO_METHOD = o3d.t.pipelines.odometry.Method.Hybrid  # depth+color → 평면 벽에서 회전 추적 안정
ODO_CRITERIA = [
    o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20),
    o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10),
    o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5),
]  # Model 기본 {6,3,1}은 회전 under-convergence. 공식 offline {20,10,5}로 상향
FITNESS_GATE = 0.3  # fitness 아래면 pose/integrate 스킵 (dead reckoning)
SURFACE_WEIGHT_THR = 3.0  # extract_point_cloud 안정화
EST_POINT_COUNT = 3000000
EXTRACT_EVERY = 10  # N프레임마다 포인트클라우드 추출 (메인 루프 여유)


def get_intrinsic_tensor(k4a, device):
    m = k4a.calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)
    # Open3D tensor intrinsic: 3x3 Float64 on CPU
    return o3c.Tensor(np.asarray(m, dtype=np.float64))


def capture_to_images(cap, device):
    """pyk4a capture → (depth_t, color_t) on `device`."""
    if cap.depth is None or cap.color is None:
        return None
    try:
        tc = cap.transformed_color  # BGRA, depth 좌표계로 정렬됨
    except RuntimeError:
        return None
    if tc is None:
        return None

    color_rgb = np.ascontiguousarray(tc[:, :, (2, 1, 0)].astype(np.uint8))
    depth = np.ascontiguousarray(cap.depth.astype(np.uint16))

    depth_t = o3d.t.geometry.Image(o3c.Tensor(depth)).to(device)
    color_t = o3d.t.geometry.Image(o3c.Tensor(color_rgb)).to(device)
    return depth_t, color_t


def main():
    server = viser.ViserServer(port=8080)
    server.scene.world_axes.visible = True
    server.scene.add_grid("/grid", width=10.0, height=10.0, plane="xz")
    server.scene.set_up_direction((0.0, 1.0, 0.0))

    gui_point_size = server.gui.add_slider("Point Size", 0.001, 0.02, 0.001, 0.005)
    gui_reset = server.gui.add_button("Reset")
    gui_status = server.gui.add_text("Status", initial_value="Starting...")
    reset_flag = [False]

    @gui_reset.on_click
    def _(_): reset_flag[0] = True

    @server.on_client_connect
    def on_connect(c):
        c.camera.position = np.array([0.0, 2.0, -3.0])
        c.camera.look_at = np.array([0.0, 0.0, 2.0])
        c.camera.up_direction = np.array([0.0, 1.0, 0.0])

    # Kinect
    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        camera_fps=pyk4a.FPS.FPS_15,
        depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    time.sleep(1.0)

    # CUDA tensor device
    device = o3c.Device("CUDA:0")
    intrinsic_t = get_intrinsic_tensor(k4a, device)
    fx = float(intrinsic_t[0, 0].item())
    print(f"Intrinsic fx={fx:.1f}, device={device}", flush=True)

    # warm up
    for _ in range(10):
        try: k4a.get_capture(timeout=2000)
        except: time.sleep(0.3)

    # depth size
    depth_h, depth_w = 512, 512  # WFOV_2X2BINNED

    def make_model_and_frames():
        T_init = o3c.Tensor(np.identity(4), o3c.Dtype.Float64)
        model = o3d.t.pipelines.slam.Model(
            VOXEL_SIZE, BLOCK_RES, BLOCK_COUNT, T_init, device,
        )
        input_frame = o3d.t.pipelines.slam.Frame(depth_h, depth_w, intrinsic_t, device)
        raycast_frame = o3d.t.pipelines.slam.Frame(depth_h, depth_w, intrinsic_t, device)
        return model, input_frame, raycast_frame

    model, input_frame, raycast_frame = make_model_and_frames()
    T_frame_to_model = o3c.Tensor(np.identity(4), o3c.Dtype.Float64)

    flip = np.diag([-1.0, -1.0, 1.0, 1.0])  # Kinect → viser(Y-up)
    frame_id = 0
    ok = fail = 0
    last_fitness = 0.0

    try:
        while True:
            try: cap = k4a.get_capture(timeout=1000)
            except: continue

            if reset_flag[0]:
                del model, input_frame, raycast_frame  # VRAM 반환
                model, input_frame, raycast_frame = make_model_and_frames()
                T_frame_to_model = o3c.Tensor(np.identity(4), o3c.Dtype.Float64)
                frame_id = 0
                ok = fail = 0
                reset_flag[0] = False

            imgs = capture_to_images(cap, device)
            if imgs is None: continue
            depth_t, color_t = imgs

            input_frame.set_data_from_image("depth", depth_t)
            input_frame.set_data_from_image("color", color_t)

            if frame_id > 0:
                try:
                    # 공식 dense_slam.py 방식: fitness gate 없이 항상 통합
                    # (빠른 회전 시 프레임 스킵하면 새 영역이 영영 누락됨)
                    result = model.track_frame_to_model(
                        input_frame, raycast_frame,
                        DEPTH_SCALE, DEPTH_MAX, ODO_DIST_THR,
                        ODO_METHOD, ODO_CRITERIA,
                    )
                    last_fitness = float(result.fitness)
                    # fitness 낮으면 잘못된 pose로 integrate → smear. 이 프레임은 통째로 스킵.
                    if last_fitness < FITNESS_GATE:
                        fail += 1
                        frame_id += 1
                        continue
                    T_frame_to_model = T_frame_to_model @ result.transformation
                    ok += 1
                except Exception as e:
                    # Singular matrix 등 ICP 자체 실패 — 이전 pose 유지 + 통합 스킵
                    print(f"  [track err f{frame_id}] {type(e).__name__}: {e}", flush=True)
                    fail += 1
                    frame_id += 1
                    continue
            else:
                ok += 1
                last_fitness = 1.0

            try:
                model.update_frame_pose(frame_id, T_frame_to_model)
                model.integrate(input_frame, DEPTH_SCALE, DEPTH_MAX, TRUNC_MULT)
                model.synthesize_model_frame(
                    raycast_frame, DEPTH_SCALE, DEPTH_MIN, DEPTH_MAX, TRUNC_MULT, True,
                )
            except Exception as e:
                print(f"  [integrate err f{frame_id}] {type(e).__name__}: {e}", flush=True)
                # frame_id는 여기서도 증가시켜야 다음 update_frame_pose와 충돌 없음
            frame_id += 1

            # N프레임마다 추출 (voxel_grid.extract_point_cloud: weight threshold로 노이즈 제거)
            if frame_id % EXTRACT_EVERY == 0:
                pcd_t = model.voxel_grid.extract_point_cloud(
                    SURFACE_WEIGHT_THR, EST_POINT_COUNT,
                ).to(o3c.Device("CPU:0"))
                pcd = pcd_t.to_legacy()
                pcd.transform(flip)
                pts = np.asarray(pcd.points, dtype=np.float32)
                if pcd.has_colors():
                    cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
                else:
                    cols = np.full((len(pts), 3), 200, dtype=np.uint8)

                if len(pts) > 0:
                    if len(pts) > 500000:
                        idx = np.random.choice(len(pts), 500000, replace=False)
                        pts, cols = pts[idx], cols[idx]
                    server.scene.add_point_cloud(
                        "/kinect", points=pts, colors=cols,
                        point_size=gui_point_size.value, point_shape="rounded",
                    )
                    gui_status.value = f"{len(pts)} pts | ok={ok} fail={fail} fit={last_fitness:.2f}"

                if frame_id % 30 == 0:
                    print(f"[{frame_id}] {len(pts)} pts, ok={ok} fail={fail} last_fit={last_fitness:.3f}", flush=True)

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()


if __name__ == "__main__":
    main()
