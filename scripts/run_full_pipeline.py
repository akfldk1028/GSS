"""Run the full GSS pipeline for specific scenes.

Usage:
    python scripts/run_full_pipeline.py                    # Both bonsai + bicycle
    python scripts/run_full_pipeline.py --scene bonsai     # Interior only
    python scripts/run_full_pipeline.py --scene bicycle    # Exterior only
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

app = typer.Typer(name="run_full_pipeline")
console = Console()
logger = logging.getLogger("gss.run_full")

# Scene registry
SCENES = {
    # --- Good building scenes ---
    "room": {
        "ply": "data/raw/inria_pretrained/room/point_cloud/iteration_30000/point_cloud.ply",
        "type": "interior",
        "desc": "Room (실내, Mip-NeRF 360)",
    },
    "train": {
        "ply": "data/raw/inria_pretrained/train/point_cloud/iteration_30000/point_cloud.ply",
        "type": "exterior",
        "desc": "Train Station (실외, Tanks & Temples)",
    },
    # --- Other scenes ---
    "bonsai": {
        "ply": "data/raw/bonsai/point_cloud.ply",
        "type": "interior",
        "desc": "Bonsai (실내, Mip-NeRF 360)",
    },
    "bicycle": {
        "ply": "data/raw/bicycle/point_cloud.ply",
        "type": "exterior",
        "desc": "Bicycle (실외, Mip-NeRF 360)",
    },
    "kitchen": {
        "ply": "data/raw/inria_pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply",
        "type": "interior",
        "desc": "Kitchen (실내, Mip-NeRF 360)",
    },
    "drjohnson": {
        "ply": "data/raw/inria_pretrained/drjohnson/point_cloud/iteration_30000/point_cloud.ply",
        "type": "interior",
        "desc": "Dr Johnson (실내, Deep Blending)",
    },
    "truck": {
        "ply": "data/raw/inria_pretrained/truck/point_cloud/iteration_30000/point_cloud.ply",
        "type": "exterior",
        "desc": "Truck (실외, Tanks & Temples)",
    },
    "counter": {
        "ply": "data/raw/inria_pretrained/counter/point_cloud/iteration_30000/point_cloud.ply",
        "type": "interior",
        "desc": "Counter (실내, Mip-NeRF 360)",
    },
    "playroom": {
        "ply": "data/raw/inria_pretrained/playroom/point_cloud/iteration_30000/point_cloud.ply",
        "type": "interior",
        "desc": "Playroom (실내, Deep Blending)",
    },
}


def _load_yaml_config(config_path: Path, config_class):
    """Load a YAML config file into a Pydantic model."""
    import yaml

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return config_class(**raw)


def run_scene(scene_name: str, scene_info: dict) -> dict:
    """Run the full pipeline for a single scene.

    Interior: s00 → s06 → s06b → s07 → s08
    Exterior: s00 → s06 → s06c → s06b → s07 → s08

    Uses scene-specific data_root to isolate outputs.
    """
    ply_path = PROJECT_ROOT / scene_info["ply"]
    scene_type = scene_info["type"]
    data_root = PROJECT_ROOT / "data" / "runs" / scene_name

    console.print(Panel(
        f"[bold]{scene_info['desc']}[/bold]\n"
        f"Type: {scene_type}\n"
        f"PLY: {ply_path}\n"
        f"Output: {data_root}",
        title=f"Scene: {scene_name}",
        border_style="cyan",
    ))

    if not ply_path.exists():
        console.print(f"[red]PLY not found: {ply_path}[/red]")
        return {"error": "PLY not found"}

    results = {}
    t_total = time.time()

    # ── Step 0: Import PLY ──────────────────────────────────────
    console.print("\n[cyan]━━━ s00: Import PLY ━━━[/cyan]")
    from gss.steps.s00_import_ply.step import ImportPlyStep
    from gss.steps.s00_import_ply.contracts import ImportPlyInput
    from gss.steps.s00_import_ply.config import ImportPlyConfig

    s00_config = _load_yaml_config(
        PROJECT_ROOT / "configs/steps/s00_import_ply.yaml", ImportPlyConfig
    )
    s00 = ImportPlyStep(config=s00_config, data_root=data_root)
    s00_out = s00.execute(ImportPlyInput(ply_path=ply_path))
    console.print(f"  [green]OK[/green] {s00_out.num_surface_points:,} points")
    results["s00"] = s00_out.model_dump()

    # ── Step 6: Plane Extraction ────────────────────────────────
    console.print("\n[cyan]━━━ s06: Plane Extraction ━━━[/cyan]")
    from gss.steps.s06_plane_extraction.step import PlaneExtractionStep
    from gss.steps.s06_plane_extraction.contracts import PlaneExtractionInput
    from gss.steps.s06_plane_extraction.config import PlaneExtractionConfig

    s06_config = _load_yaml_config(
        PROJECT_ROOT / "configs/steps/s06_plane_extraction.yaml", PlaneExtractionConfig
    )
    s06 = PlaneExtractionStep(config=s06_config, data_root=data_root)
    s06_input = PlaneExtractionInput(
        surface_points_path=s00_out.surface_points_path,
        metadata_path=s00_out.metadata_path,
    )
    s06_out = s06.execute(s06_input)
    console.print(
        f"  [green]OK[/green] {s06_out.num_planes} planes "
        f"(walls={s06_out.num_walls}, floors={s06_out.num_floors}, "
        f"ceilings={s06_out.num_ceilings})"
    )
    results["s06"] = s06_out.model_dump()

    # ── Step 6c: Building Extraction (exterior only) ────────────
    planes_for_s06b = s06_out.planes_file
    boundaries_for_s06b = s06_out.boundaries_file

    if scene_type == "exterior":
        console.print("\n[cyan]━━━ s06c: Building Extraction ━━━[/cyan]")
        from gss.steps.s06c_building_extraction.step import BuildingExtractionStep
        from gss.steps.s06c_building_extraction.contracts import BuildingExtractionInput
        from gss.steps.s06c_building_extraction.config import BuildingExtractionConfig

        s06c_config = _load_yaml_config(
            PROJECT_ROOT / "configs/steps/s06c_building_extraction.yaml",
            BuildingExtractionConfig,
        )
        s06c = BuildingExtractionStep(config=s06c_config, data_root=data_root)
        s06c_input = BuildingExtractionInput(
            planes_file=s06_out.planes_file,
            boundaries_file=s06_out.boundaries_file,
        )
        s06c_out = s06c.execute(s06c_input)
        console.print(
            f"  [green]OK[/green] facades={s06c_out.num_facades}, "
            f"roof_faces={s06c_out.num_roof_faces}, "
            f"storeys={s06c_out.num_storeys}"
        )
        results["s06c"] = s06c_out.model_dump()
        planes_for_s06b = s06c_out.planes_file
        boundaries_for_s06b = s06c_out.boundaries_file

    # ── Step 6b: Plane Regularization ───────────────────────────
    console.print("\n[cyan]━━━ s06b: Plane Regularization ━━━[/cyan]")
    from gss.steps.s06b_plane_regularization.step import PlaneRegularizationStep
    from gss.steps.s06b_plane_regularization.contracts import PlaneRegularizationInput
    from gss.steps.s06b_plane_regularization.config import PlaneRegularizationConfig

    s06b_config = _load_yaml_config(
        PROJECT_ROOT / "configs/steps/s06b_plane_regularization.yaml",
        PlaneRegularizationConfig,
    )
    # For exterior, use cluster normal mode
    if scene_type == "exterior":
        s06b_config = s06b_config.model_copy(update={"normal_mode": "cluster"})

    s06b = PlaneRegularizationStep(config=s06b_config, data_root=data_root)
    s06b_input = PlaneRegularizationInput(
        planes_file=planes_for_s06b,
        boundaries_file=boundaries_for_s06b,
    )
    s06b_out = s06b.execute(s06b_input)
    console.print(
        f"  [green]OK[/green] walls={s06b_out.num_walls}, "
        f"spaces={s06b_out.num_spaces}"
    )
    results["s06b"] = s06b_out.model_dump()

    # ── Step 7: IFC Export ──────────────────────────────────────
    console.print("\n[cyan]━━━ s07: IFC Export ━━━[/cyan]")
    from gss.steps.s07_ifc_export.step import IfcExportStep
    from gss.steps.s07_ifc_export.contracts import IfcExportInput
    from gss.steps.s07_ifc_export.config import IfcExportConfig

    s07_config = _load_yaml_config(
        PROJECT_ROOT / "configs/steps/s07_ifc_export.yaml", IfcExportConfig
    )
    # Set project name per scene
    s07_config = s07_config.model_copy(update={
        "project_name": f"GSS_{scene_name}",
        "building_name": scene_info["desc"].split("(")[0].strip(),
    })

    s07 = IfcExportStep(config=s07_config, data_root=data_root)
    s07_input = IfcExportInput(
        walls_file=s06b_out.walls_file,
        spaces_file=s06b_out.spaces_file,
        planes_file=s06b_out.planes_file,
    )
    s07_out = s07.execute(s07_input)
    console.print(
        f"  [green]OK[/green] walls={s07_out.num_walls}, "
        f"slabs={s07_out.num_slabs}, spaces={s07_out.num_spaces}, "
        f"openings={s07_out.num_openings}"
    )
    console.print(f"  IFC: {s07_out.ifc_path}")
    results["s07"] = s07_out.model_dump()

    # ── Step 8: Mesh Export ─────────────────────────────────────
    console.print("\n[cyan]━━━ s08: Mesh Export ━━━[/cyan]")
    from gss.steps.s08_mesh_export.step import MeshExportStep
    from gss.steps.s08_mesh_export.contracts import MeshExportInput
    from gss.steps.s08_mesh_export.config import MeshExportConfig

    s08_config = _load_yaml_config(
        PROJECT_ROOT / "configs/steps/s08_mesh_export.yaml", MeshExportConfig
    )
    s08 = MeshExportStep(config=s08_config, data_root=data_root)
    s08_input = MeshExportInput(ifc_path=s07_out.ifc_path)
    s08_out = s08.execute(s08_input)
    console.print(
        f"  [green]OK[/green] meshes={s08_out.num_meshes}, "
        f"verts={s08_out.num_vertices:,}, faces={s08_out.num_faces:,}"
    )
    if s08_out.glb_path:
        console.print(f"  GLB: {s08_out.glb_path}")
    if s08_out.usd_path:
        console.print(f"  USD: {s08_out.usd_path}")
    results["s08"] = s08_out.model_dump()

    elapsed = time.time() - t_total
    console.print(f"\n[bold green]Done: {scene_name} ({elapsed:.1f}s)[/bold green]")

    # Save run summary
    summary_path = data_root / "processed" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"scene": scene_name, "type": scene_type, "elapsed_s": elapsed, **results},
            f, indent=2, default=_serialize,
        )

    return results


@app.command()
def main(
    scene: str = typer.Option(
        None, "--scene", "-s",
        help="Scene name (bonsai, bicycle). Omit for both.",
    ),
) -> None:
    """Run full GSS pipeline end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if scene:
        if scene not in SCENES:
            console.print(f"[red]Unknown scene: {scene}[/red]")
            console.print(f"Available: {', '.join(SCENES.keys())}")
            raise typer.Exit(1)
        scenes_to_run = {scene: SCENES[scene]}
    else:
        scenes_to_run = SCENES

    all_results = {}
    for name, info in scenes_to_run.items():
        try:
            all_results[name] = run_scene(name, info)
        except Exception as e:
            console.print(f"\n[red]FAILED: {name}[/red]")
            console.print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # ── Summary ────────────────────────────────────────────────
    console.print("\n")
    table = Table(title="Pipeline Results Summary")
    table.add_column("Scene", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Planes", style="green")
    table.add_column("Walls", style="green")
    table.add_column("Spaces", style="green")
    table.add_column("Meshes", style="green")
    table.add_column("Status", style="bold")

    for name, res in all_results.items():
        if "error" in res:
            table.add_row(name, SCENES[name]["type"], "-", "-", "-", "-", f"[red]{res['error']}[/red]")
        else:
            s06 = res.get("s06", {})
            s06b = res.get("s06b", {})
            s08 = res.get("s08", {})
            table.add_row(
                name,
                SCENES[name]["type"],
                str(s06.get("num_planes", "?")),
                str(s06b.get("num_walls", "?")),
                str(s06b.get("num_spaces", "?")),
                str(s08.get("num_meshes", "?")),
                "[green]OK[/green]",
            )

    console.print(table)

    # Output locations
    console.print("\n[bold]Output files:[/bold]")
    for name in all_results:
        if "error" not in all_results[name]:
            base = PROJECT_ROOT / "data" / "runs" / name / "processed"
            console.print(f"  {name}:")
            for f in sorted(base.glob("*")) if base.exists() else []:
                size_mb = f.stat().st_size / (1024 * 1024)
                console.print(f"    {f.name} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    app()
