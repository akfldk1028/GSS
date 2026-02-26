"""Verify all PlanarGS dependencies are importable."""
import torch
print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
import simple_knn
print("simple_knn OK")
import diff_plane_rasterization
print("diff_plane_rasterization OK")
from pytorch3d.transforms import quaternion_to_matrix
print("pytorch3d OK")
import groundingdino
print("groundingdino OK")
import segment_anything
print("segment_anything OK")
import open3d
print(f"open3d={open3d.__version__}")
print("ALL IMPORTS OK")
