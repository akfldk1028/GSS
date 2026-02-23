# S07: IFC/BIM Export

Generate IFC4 BIM file from detected planes and boundaries.

> **Note**: After s06 v4 improvements (Manhattan alignment + coplanar merge + position filtering),
> typical input is 3 walls + 1 floor + 2 ceilings with clean rectangular boundaries.

## Input
| Field | Type | Description |
|-------|------|-------------|
| `planes_file` | Path | planes.json from S06 |
| `boundaries_file` | Path | boundaries.json from S06 |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `ifc_path` | Path | Generated .ifc file |
| `num_walls` | int | IfcWall objects |
| `num_slabs` | int | IfcSlab objects |
| `ifc_version` | str | IFC schema version |

## IFC Hierarchy
```
IfcProject
  └─ IfcSite
      └─ IfcBuilding
          └─ IfcBuildingStorey
              ├─ IfcWall (from wall planes)
              └─ IfcSlab (from floor/ceiling planes)
```

## Config (`configs/steps/s07_ifc_export.yaml`)
| Key | Default | Description |
|-----|---------|-------------|
| `ifc_version` | IFC4 | Schema version |
| `project_name` | GSS_BIM | IFC project name |
| `building_name` | Building | Building name |
| `default_wall_thickness` | 0.2 | Wall thickness (m) |
| `default_slab_thickness` | 0.3 | Slab thickness (m) |

## Dependencies
- `ifcopenshell >= 0.7`
