"""JH closeup overlay — big single figure."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
import open3d as o3d

RUNS = 'data/runs/jh'
IMAGE = 'docs/images/jh_overlay_closeup.png'

pcd = o3d.io.read_point_cloud(f'{RUNS}/interim/s00_import_ply/surface_points.ply')
pts = np.asarray(pcd.points)
np.random.seed(42)
pts = pts[np.random.choice(len(pts), 80000, replace=False)]

align = json.load(open(f'{RUNS}/interim/s06b_plane_regularization/manhattan_alignment.json'))
spaces = json.load(open(f'{RUNS}/interim/s06b_plane_regularization/spaces.json'))
R = np.array(align['manhattan_rotation'])
scale = spaces['coordinate_scale']

m = pts @ R.T
ifc_pts = np.column_stack([m[:,0]/scale, m[:,2]/scale, m[:,1]/scale])

import ifcopenshell, ifcopenshell.geom
ifc = ifcopenshell.open(f'{RUNS}/processed/GSS_jh.ifc')
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)
COLORS = {'IfcWall': [0.0, 0.2, 1.0], 'IfcSlab': [0.0, 0.8, 0.0]}
all_edges = {}
for cls in ('IfcWall', 'IfcSlab'):
    for el in ifc.by_type(cls):
        try:
            shape = ifcopenshell.geom.create_shape(settings, el)
        except: continue
        v = np.array(shape.geometry.verts).reshape(-1,3)
        f = np.array(shape.geometry.faces).reshape(-1,3)
        c = tuple(COLORS[cls])
        if c not in all_edges: all_edges[c] = []
        es = set()
        for tri in f:
            for i in range(3):
                es.add(tuple(sorted([tri[i], tri[(i+1)%3]])))
        for e in es:
            all_edges[c].append((v[e[0]], v[e[1]]))

fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111, projection='3d')

z_norm = (ifc_pts[:,2] - ifc_pts[:,2].min()) / (ifc_pts[:,2].ptp()+1e-8)
ax.scatter(ifc_pts[:,0], ifc_pts[:,1], ifc_pts[:,2], c=z_norm, cmap='coolwarm', s=0.8, alpha=0.5)

for color, edges in all_edges.items():
    segs = [(list(p0), list(p1)) for p0, p1 in edges]
    ax.add_collection3d(Line3DCollection(segs, colors=[color], linewidths=3.0))

bim_all = np.vstack([np.vstack([p0,p1]) for edges in all_edges.values() for p0,p1 in edges])
bmin, bmax = bim_all.min(0), bim_all.max(0)
bc = (bmin+bmax)/2; h = max(bmax-bmin)/2; p = h*0.4
ax.set_xlim(bc[0]-h-p, bc[0]+h+p)
ax.set_ylim(bc[1]-h-p, bc[1]+h+p)
ax.set_zlim(bc[2]-h-p, bc[2]+h+p)
ax.view_init(elev=25, azim=-55)
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title('JH - 3DGS Point Cloud + BIM Wireframe Overlay\n80K pts, scale=1.70, 4W 1F 1C', fontsize=15, fontweight='bold')
ax.legend(handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',markersize=8,label='3DGS Points'),
    Line2D([0],[0],color=[0,0.2,1],lw=3,label='IfcWall'),
    Line2D([0],[0],color=[0,0.8,0],lw=3,label='IfcSlab (floor/ceiling)'),
], fontsize=12, loc='upper left')

fig.tight_layout()
fig.savefig(IMAGE, dpi=150, bbox_inches='tight')
print(f'Saved: {IMAGE}')
os.startfile(IMAGE)
