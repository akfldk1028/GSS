"""Tests for S07: IFC Export — slab, space, roof, column, tessellation builders."""

from tests.test_steps.conftest import needs_ifc


# ── Slab builder tests ──


@needs_ifc
class TestSlabBuilder:
    def test_floor_slab(self, ifc_context):
        from gss.steps.s07_ifc_export._slab_builder import create_floor_slab

        ctx = ifc_context
        space = {
            "id": 0,
            "boundary_2d": [[-1, 3], [1, 3], [1, -1], [-1, -1], [-1, 3]],
            "floor_height": -0.5,
            "ceiling_height": 2.5,
        }
        slab = create_floor_slab(ctx, space, scale=1.0, thickness=0.3)
        assert slab is not None
        assert slab.is_a("IfcSlab")
        assert slab.Name == "Floor_Room0"

    def test_ceiling_slab(self, ifc_context):
        from gss.steps.s07_ifc_export._slab_builder import create_ceiling_slab

        ctx = ifc_context
        space = {
            "id": 0,
            "boundary_2d": [[-1, 3], [1, 3], [1, -1], [-1, -1], [-1, 3]],
            "floor_height": -0.5,
            "ceiling_height": 2.5,
        }
        slab = create_ceiling_slab(ctx, space, scale=1.0, thickness=0.3)
        assert slab is not None
        assert slab.Name == "Ceiling_Room0"


# ── Space builder tests ──


@needs_ifc
class TestSpaceBuilder:
    def test_create_space(self, ifc_context):
        from gss.steps.s07_ifc_export._space_builder import create_space

        ctx = ifc_context
        space_data = {
            "id": 0,
            "boundary_2d": [[-1, 3], [1, 3], [1, -1], [-1, -1], [-1, 3]],
            "area": 8.0,
            "floor_height": -0.5,
            "ceiling_height": 2.5,
        }
        space = create_space(ctx, space_data, scale=1.0)
        assert space is not None
        assert space.is_a("IfcSpace")
        assert space.Name == "Room_0"

        # Check property set
        psets = ctx.ifc.by_type("IfcPropertySet")
        pset_names = [ps.Name for ps in psets]
        assert "Pset_SpaceCommon" in pset_names

    def test_space_area_scaled(self, ifc_context):
        from gss.steps.s07_ifc_export._space_builder import create_space

        ctx = ifc_context
        space_data = {
            "id": 0,
            "boundary_2d": [[-2, 6], [2, 6], [2, -2], [-2, -2], [-2, 6]],
            "area": 32.0,
            "floor_height": 0.0,
            "ceiling_height": 4.0,
        }
        create_space(ctx, space_data, scale=2.0)
        # area_m2 = 32.0 / (2.0*2.0) = 8.0
        psets = ctx.ifc.by_type("IfcPropertySet")
        for ps in psets:
            if ps.Name == "Pset_SpaceCommon":
                for prop in ps.HasProperties:
                    if prop.Name == "GrossFloorArea":
                        assert abs(prop.NominalValue.wrappedValue - 8.0) < 0.01


# ── Roof builder tests ──


@needs_ifc
class TestRoofBuilder:
    def test_flat_roof(self, ifc_context):
        from gss.steps.s07_ifc_export._roof_builder import create_roof

        ctx = ifc_context
        roof_planes = [{
            "id": 0, "label": "roof", "roof_type": "flat",
            "normal": [0.0, 1.0, 0.0], "d": -3.5,
            "boundary_3d": [[0, 3.5, 0], [5, 3.5, 0], [5, 3.5, 4], [0, 3.5, 4]],
        }]
        ifc_roof, count = create_roof(ctx, roof_planes, scale=1.0)
        assert ifc_roof is not None
        assert ifc_roof.is_a("IfcRoof")
        assert count == 1
        # Should have IfcSlab with ROOF type
        slabs = ctx.ifc.by_type("IfcSlab")
        roof_slabs = [s for s in slabs if s.PredefinedType == "ROOF"]
        assert len(roof_slabs) == 1

    def test_no_roof_planes(self, ifc_context):
        from gss.steps.s07_ifc_export._roof_builder import create_roof

        ctx = ifc_context
        ifc_roof, count = create_roof(ctx, [], scale=1.0)
        assert ifc_roof is None
        assert count == 0

    def test_multiple_roof_slabs(self, ifc_context):
        from gss.steps.s07_ifc_export._roof_builder import create_roof

        ctx = ifc_context
        roof_planes = [
            {"id": 0, "label": "roof", "roof_type": "flat",
             "normal": [0.0, 1.0, 0.0], "d": -3.5,
             "boundary_3d": [[0, 3.5, 0], [5, 3.5, 0], [5, 3.5, 4], [0, 3.5, 4]]},
            {"id": 1, "label": "roof", "roof_type": "flat",
             "normal": [0.0, 1.0, 0.0], "d": -4.0,
             "boundary_3d": [[0, 4.0, 0], [3, 4.0, 0], [3, 4.0, 2], [0, 4.0, 2]]},
        ]
        _, count = create_roof(ctx, roof_planes, scale=1.0)
        assert count == 2


# ── Structured roof tests ──


@needs_ifc
class TestStructuredRoof:
    def test_structured_roof_gable(self, ifc_context):
        from gss.steps.s07_ifc_export._roof_builder import create_structured_roof

        ctx = ifc_context
        roof_planes = [
            {"id": 0, "label": "roof", "roof_type": "inclined",
             "normal": [0.0, 0.7, 0.7], "d": -5.0,
             "boundary_3d": [[0, 4, 0], [5, 4, 0], [5, 5, 2], [0, 5, 2]]},
            {"id": 1, "label": "roof", "roof_type": "inclined",
             "normal": [0.0, 0.7, -0.7], "d": -5.0,
             "boundary_3d": [[0, 4, 4], [5, 4, 4], [5, 5, 2], [0, 5, 2]]},
        ]
        roof_structure = {
            "roof_type": "gable",
            "faces": [
                {"id": 0, "plane_id": 0, "slope_deg": 45.0, "aspect": "south", "sub_type": "inclined"},
                {"id": 1, "plane_id": 1, "slope_deg": 45.0, "aspect": "north", "sub_type": "inclined"},
            ],
            "ridges": [[[0, 5, 2], [5, 5, 2]]],
            "eaves": [],
            "valleys": [],
        }
        ifc_roof, count, roof_type = create_structured_roof(
            ctx, roof_planes, roof_structure, scale=1.0,
        )
        assert ifc_roof is not None
        assert ifc_roof.PredefinedType == "GABLE_ROOF"
        assert count == 2
        assert roof_type == "gable"

    def test_roof_ridge_annotation(self, ifc_context):
        from gss.steps.s07_ifc_export._roof_builder import create_structured_roof

        ctx = ifc_context
        roof_planes = [
            {"id": 0, "label": "roof", "roof_type": "flat",
             "normal": [0.0, 1.0, 0.0], "d": -4.0,
             "boundary_3d": [[0, 4, 0], [5, 4, 0], [5, 4, 4], [0, 4, 4]]},
        ]
        roof_structure = {
            "roof_type": "flat",
            "faces": [],
            "ridges": [[[0, 4, 2], [5, 4, 2]]],
            "eaves": [[[0, 4, 0], [5, 4, 0]]],
            "valleys": [],
        }
        create_structured_roof(
            ctx, roof_planes, roof_structure, scale=1.0,
            create_annotations=True,
        )
        annotations = ctx.ifc.by_type("IfcAnnotation")
        assert len(annotations) == 2  # 1 ridge + 1 eave

    def test_roof_fallback_no_context(self, ifc_context):
        """Without roof_structure, create_roof() (basic) should work."""
        from gss.steps.s07_ifc_export._roof_builder import create_roof

        ctx = ifc_context
        roof_planes = [{
            "id": 0, "label": "roof", "roof_type": "flat",
            "normal": [0.0, 1.0, 0.0], "d": -3.5,
            "boundary_3d": [[0, 3.5, 0], [5, 3.5, 0], [5, 3.5, 4], [0, 3.5, 4]],
        }]
        ifc_roof, count = create_roof(ctx, roof_planes, scale=1.0)
        assert ifc_roof is not None
        assert count == 1
        # No PredefinedType set in basic mode
        assert ifc_roof.PredefinedType is None

    def test_roof_pset(self, ifc_context):
        from gss.steps.s07_ifc_export._roof_builder import create_structured_roof

        ctx = ifc_context
        roof_planes = [
            {"id": 0, "label": "roof", "roof_type": "flat",
             "normal": [0.0, 1.0, 0.0], "d": -4.0,
             "boundary_3d": [[0, 4, 0], [5, 4, 0], [5, 4, 4], [0, 4, 4]]},
        ]
        roof_structure = {
            "roof_type": "flat",
            "faces": [{"id": 0, "plane_id": 0, "slope_deg": 0.0, "aspect": "flat", "sub_type": "flat"}],
            "ridges": [], "eaves": [], "valleys": [],
        }
        create_structured_roof(ctx, roof_planes, roof_structure, scale=1.0)
        psets = ctx.ifc.by_type("IfcPropertySet")
        pset_names = [ps.Name for ps in psets]
        assert "Pset_RoofCommon" in pset_names


# ── Column builder tests ──


@needs_ifc
class TestColumnBuilder:
    def test_column_builder_circle_profile(self, ifc_context):
        from gss.steps.s07_ifc_export._column_builder import create_column

        ctx = ifc_context
        col_data = {
            "id": 0, "column_type": "round",
            "center_2d": [5.2, 3.1], "radius": 0.15,
            "height_range": [0.0, 2.7],
        }
        col = create_column(ctx, col_data, scale=1.0)
        assert col is not None
        assert col.is_a("IfcColumn")
        profiles = ctx.ifc.by_type("IfcCircleProfileDef")
        assert len(profiles) == 1
        assert abs(profiles[0].Radius - 0.15) < 0.001

    def test_column_builder_rectangle_profile(self, ifc_context):
        from gss.steps.s07_ifc_export._column_builder import create_column

        ctx = ifc_context
        col_data = {
            "id": 1, "column_type": "rectangular",
            "center_2d": [2.0, 1.0], "width": 0.4, "depth": 0.2,
            "height_range": [0.0, 3.0],
            "direction": [1.0, 0.0],
        }
        col = create_column(ctx, col_data, scale=1.0)
        assert col is not None
        profiles = ctx.ifc.by_type("IfcRectangleProfileDef")
        assert len(profiles) == 1
        assert abs(profiles[0].XDim - 0.4) < 0.001
        assert abs(profiles[0].YDim - 0.2) < 0.001

    def test_column_type_created(self, ifc_context):
        from gss.steps.s07_ifc_export._column_builder import create_column

        ctx = ifc_context
        col_data = {
            "id": 0, "column_type": "round",
            "center_2d": [1.0, 1.0], "radius": 0.1,
            "height_range": [0.0, 3.0],
        }
        create_column(ctx, col_data, scale=1.0)
        col_types = ctx.ifc.by_type("IfcColumnType")
        assert len(col_types) >= 1
        assert col_types[0].PredefinedType == "COLUMN"


# ── Tessellation builder tests ──


@needs_ifc
class TestTessellationBuilder:
    def test_tessellation_cube(self, ifc_context):
        """8-vertex 12-face cube → IfcPolygonalFaceSet."""
        from gss.steps.s07_ifc_export._tessellation_builder import create_tessellated_element

        ctx = ifc_context
        # Simple cube vertices
        verts = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
        ]
        # 12 triangular faces
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ]
        element = create_tessellated_element(
            ctx, verts, faces,
            ifc_class="IfcBuildingElementProxy",
            name="TestCube",
            scale=1.0,
        )
        assert element is not None
        assert element.is_a("IfcBuildingElementProxy")
        assert element.Name == "TestCube"

        face_sets = ctx.ifc.by_type("IfcPolygonalFaceSet")
        assert len(face_sets) == 1
        assert len(face_sets[0].Faces) == 12

    def test_tessellation_representation_type(self, ifc_context):
        from gss.steps.s07_ifc_export._tessellation_builder import create_tessellated_element

        ctx = ifc_context
        verts = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]
        faces = [[0, 1, 2]]
        element = create_tessellated_element(ctx, verts, faces, scale=1.0)
        assert element is not None
        rep = element.Representation.Representations[0]
        assert rep.RepresentationType == "Tessellation"

    def test_tessellation_1based_index(self, ifc_context):
        """IFC uses 1-based indexing for face indices."""
        from gss.steps.s07_ifc_export._tessellation_builder import create_tessellated_element

        ctx = ifc_context
        verts = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]
        faces = [[0, 1, 2]]  # 0-based input
        create_tessellated_element(ctx, verts, faces, scale=1.0)
        face_sets = ctx.ifc.by_type("IfcPolygonalFaceSet")
        ifc_face = face_sets[0].Faces[0]
        # Should be 1-based: [1, 2, 3]
        assert list(ifc_face.CoordIndex) == [1, 2, 3]

    def test_tessellation_with_color(self, ifc_context):
        from gss.steps.s07_ifc_export._tessellation_builder import create_tessellated_element

        ctx = ifc_context
        verts = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]
        faces = [[0, 1, 2]]
        element = create_tessellated_element(
            ctx, verts, faces, scale=1.0, color=[0.8, 0.2, 0.2],
        )
        assert element is not None
        styled = ctx.ifc.by_type("IfcStyledItem")
        assert len(styled) >= 1
