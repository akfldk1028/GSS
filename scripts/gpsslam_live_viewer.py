"""GPS-SLAM 결과 실시간 웹 뷰어. http://localhost:8080

tsdf_mesh.ply 또는 point_cloud.ply를 viser로 표시.
--live 모드: Kinect 캡처 → GPS-SLAM → 결과 갱신 루프.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import viser


def load_mesh(path: str) -> tuple:
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.uint32)
    if mesh.has_vertex_colors():
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    else:
        colors = np.full((len(verts), 3), 180, dtype=np.uint8)
    return verts, faces, colors


def load_pointcloud(path: str) -> tuple:
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float32)
    if pcd.has_colors():
        cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        cols = np.full((len(pts), 3), 180, dtype=np.uint8)
    return pts, cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default=None, help="Path to tsdf_mesh.ply")
    parser.add_argument("--pcd", type=str, default=None, help="Path to point_cloud.ply")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.mesh is None and args.pcd is None:
        # Default: GPS-SLAM output
        base = Path("clone/GPS-SLAM/output/kinect_live")
        if (base / "tsdf_mesh.ply").exists():
            args.mesh = str(base / "tsdf_mesh.ply")
        if (base / "gs_model/point_cloud/point_cloud.ply").exists():
            args.pcd = str(base / "gs_model/point_cloud/point_cloud.ply")

    server = viser.ViserServer(port=args.port)
    print(f"Viewer: http://localhost:{args.port}")

    if args.mesh:
        print(f"Loading mesh: {args.mesh}")
        verts, faces, colors = load_mesh(args.mesh)
        server.scene.add_mesh_simple(
            "/mesh",
            vertices=verts,
            faces=faces,
            color=(180, 180, 180),
            wireframe=False,
        )
        print(f"Mesh: {len(verts)} vertices, {len(faces)} faces")

    if args.pcd:
        print(f"Loading pointcloud: {args.pcd}")
        pts, cols = load_pointcloud(args.pcd)
        server.scene.add_point_cloud(
            "/gaussians",
            points=pts,
            colors=cols,
            point_size=0.005,
            point_shape="rounded",
        )
        print(f"Points: {len(pts)}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
