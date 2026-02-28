# S07: IFC/BIM Export

Generate IFC4 BIM file from center-line wall data (Cloud2BIM pattern).

## Input
| Field | Type | Description |
|-------|------|-------------|
| `walls_file` | Path | walls.json from S06b (center-lines + thickness) |
| `spaces_file` | Path | spaces.json from S06b (room polygons, optional) |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `ifc_path` | Path | Generated .ifc file |
| `num_walls` | int | IfcWall objects |
| `num_slabs` | int | IfcSlab objects |
| `num_spaces` | int | IfcSpace objects |
| `num_openings` | int | IfcOpeningElement objects (doors + windows) |
| `ifc_version` | str | IFC schema version |

## IFC Hierarchy
```
IfcProject → IfcSite → IfcBuilding → IfcBuildingStorey
  ├─ IfcWall (center_line_2d → extruded rectangle)
  │   └─ IfcRelVoidsElement → IfcOpeningElement
  │        └─ IfcRelFillsElement → IfcDoor / IfcWindow
  ├─ IfcSlab (floor + ceiling from boundary_2d)
  └─ IfcSpace (room polygons, aggregated via IfcRelAggregates)
```

## Sub-modules
| Module | Description |
|--------|-------------|
| `_ifc_builder.py` | Project/Site/Building/Storey hierarchy + contexts |
| `_wall_builder.py` | Center-line → IfcWall (extruded rectangle profile) |
| `_slab_builder.py` | Floor/ceiling IfcSlab from boundary_2d |
| `_space_builder.py` | IfcSpace from room polygons |
| `_opening_builder.py` | IfcOpeningElement + IfcDoor/IfcWindow from openings data. Opening z is relative to wall placement (not absolute). Fill elements get distinct geometry. |

## Config (`configs/steps/s07_ifc_export.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `ifc_version` | IFC4 | Schema version |
| `project_name` | GSS_BIM | IFC project name |
| `building_name` | Building | Building name |
| `default_wall_thickness` | 0.2 | Wall thickness (m) |
| `default_slab_thickness` | 0.3 | Slab thickness (m) |
| `include_synthetic_walls` | true | Include synthetic walls in export |
| `create_spaces` | true | Create IfcSpace objects |

## Dependencies
- `ifcopenshell >= 0.7`
