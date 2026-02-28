# S08: Mesh Export (IFC → GLB / USD)

Export IFC geometry to universal 3D formats for digital twin platforms.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `ifc_path` | Path | .ifc file from S07 |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `glb_path` | Path | Exported .glb file |
| `usd_path` | Path | Exported .usdc file |
| `usdz_path` | Path | Exported .usdz file (optional) |
| `num_meshes` | int | Total mesh count |
| `num_vertices` | int | Total vertex count |
| `num_faces` | int | Total face count |

## Formats
- **GLB** — Universal glTF Binary. Supported by all major 3D viewers, UE5, Unity, Three.js, Blender.
- **USDC** — Binary USD. Required for NVIDIA Omniverse and Isaac Sim.
- **USDZ** — Packaged USD. Required for Apple Vision Pro / ARKit.

## Config (`configs/steps/s08_mesh_export.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `export_glb` | true | Export GLB file |
| `export_usd` | true | Export USDC file |
| `export_usdz` | false | Export USDZ package |
| `color_scheme` | by_class | Color by IFC class |
| `include_spaces` | false | Include IfcSpace geometry |
| `usd_up_axis` | Z | USD up axis ("Y" or "Z") |
| `usd_meters_per_unit` | 1.0 | USD scale factor |

## Coordinate Transform (Z-up → Y-up)

IFC uses **Z-up** convention. GLB (glTF) and USD with `up_axis=Y` use **Y-up**.

The transform applied: `(x, y, z) → (x, z, -y)`

| Source | Target | Transform |
|--------|--------|-----------|
| IFC Z-up | GLB Y-up | Always applied |
| IFC Z-up | USD Z-up | No transform (default) |
| IFC Z-up | USD Y-up | Same as GLB (`usd_up_axis: "Y"`) |

## Graceful Degradation

Both `trimesh` and `usd-core` are checked at runtime:
- **trimesh** missing → GLB export skipped with warning
- **usd-core** missing → USD/USDZ export skipped with warning

## USD Details
- Materials: `UsdPreviewSurface` PBR (roughness=0.7, metallic=0.0, per-class diffuseColor)
- `MaterialBindingAPI.Apply()` called before `Bind()` (Omniverse spec compliance)
- Material deduplication by color key (RGBA)
- Face vertex counts: all triangles (`[3] * n_faces`)

## Visualization
```bash
# Single file
python scripts/visualize_mesh.py data/processed/GSS_BIM.glb
python scripts/visualize_mesh.py data/processed/GSS_BIM.usdc

# Compare GLB vs USD side by side
python scripts/visualize_mesh.py --compare --save

# Save PNG renders
python scripts/visualize_mesh.py data/processed/GSS_BIM.glb --save
```

Output: `docs/images/mesh_*_isometric.png`, `mesh_*_topdown.png`

## Dependencies
- `ifcopenshell >= 0.7` — IFC geometry extraction
- `trimesh >= 4.0` — GLB export (graceful skip if missing)
- `usd-core >= 24.0` (optional) — USD/USDZ export (graceful skip if missing)
- Note: `usd-core 24.11` recommended on Windows (26.x has DLL init issues)
