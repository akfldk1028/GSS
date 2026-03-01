"""Download public 3DGS datasets for GSS pipeline testing.

Supports pre-trained 3DGS PLY files (direct import via s00) and
COLMAP+images datasets (full pipeline s01-s08).

Usage:
    python scripts/download_sample_data.py list
    python scripts/download_sample_data.py download -d bonsai
    python scripts/download_sample_data.py download -d tandt_train
    python scripts/download_sample_data.py download -d mipnerf360
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)
from rich.table import Table

app = typer.Typer(name="download_sample_data", help="Download 3DGS datasets for GSS pipeline")
console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# HuggingFace individual 3DGS scenes (pre-trained PLY, ~50-200MB each)
_HF_3DGS_BASE = "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main"
_HF_SCENES = {
    # Interior (PLY verified available)
    "bonsai":  {"type": "interior", "desc": "Bonsai tree on table (Mip-NeRF 360, ~294MB)"},
    # Exterior / outdoor (PLY verified available)
    "bicycle": {"type": "exterior", "desc": "Bicycle outdoors (Mip-NeRF 360, ~500MB)"},
    "stump":   {"type": "exterior", "desc": "Tree stump outdoors (Mip-NeRF 360, ~400MB)"},
}

# Build per-scene entries
for scene_name, info in _HF_SCENES.items():
    pass  # Will be dynamically generated in DATASETS below

# INRIA pre-trained models (all 13 scenes in one zip, ~14GB)
_INRIA_MODELS_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"

# Tanks & Temples + Deep Blending COLMAP input (~650MB)
_TANDT_DB_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"

# Mip-NeRF 360 images + COLMAP (~3-5GB)
_MIPNERF360_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


DATASETS: dict[str, dict] = {}

# --- Individual HuggingFace 3DGS scenes (small, fast) ---
for scene_name, info in _HF_SCENES.items():
    DATASETS[scene_name] = {
        "name": f"3DGS {scene_name} ({info['type']})",
        "desc": info["desc"],
        "url": f"{_HF_3DGS_BASE}/{scene_name}/point_cloud/iteration_30000/point_cloud.ply",
        "target_dir": f"data/raw/{scene_name}",
        "download_type": "ply",  # single PLY file
        "scene_type": info["type"],
        "size_hint": "50-200MB",
    }

# --- INRIA pre-trained models (all scenes, large) ---
DATASETS["inria_pretrained"] = {
    "name": "INRIA Pre-trained 3DGS (13 scenes)",
    "desc": "All Mip-NeRF360 + T&T + DeepBlending scenes. ~14GB zip.",
    "url": _INRIA_MODELS_URL,
    "target_dir": "data/raw/inria_pretrained",
    "download_type": "zip",
    "scene_type": "mixed",
    "size_hint": "~14GB",
}

# --- Tanks & Temples + Deep Blending COLMAP input ---
DATASETS["tandt_db"] = {
    "name": "Tanks & Temples + Deep Blending (COLMAP)",
    "desc": "Images + COLMAP sparse. truck, train, drjohnson, playroom. ~650MB.",
    "url": _TANDT_DB_URL,
    "target_dir": "data/raw/tandt_db",
    "download_type": "zip",
    "scene_type": "exterior",
    "size_hint": "~650MB",
}

# --- Mip-NeRF 360 full dataset ---
DATASETS["mipnerf360"] = {
    "name": "Mip-NeRF 360 (images + COLMAP)",
    "desc": "9 scenes: bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill. ~3-5GB.",
    "url": _MIPNERF360_URL,
    "target_dir": "data/raw/mipnerf360",
    "download_type": "zip",
    "scene_type": "mixed",
    "size_hint": "~3-5GB",
}

# --- NeRF Synthetic Lego (legacy) ---
DATASETS["lego"] = {
    "name": "NeRF Synthetic Lego",
    "desc": "NeRF synthetic lego scene. Small test dataset.",
    "url": "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip",
    "target_dir": "data/raw/lego",
    "download_type": "zip",
    "extract_subdir": "nerf_synthetic/lego",
    "scene_type": "synthetic",
    "size_hint": "~200MB",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, output_path: Path) -> None:
    """Download a file with rich progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Google Drive
    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        try:
            import gdown
            console.print(f"[cyan]Downloading from Google Drive...[/cyan]")
            gdown.download(url, str(output_path), quiet=False, fuzzy=True)
            return
        except ImportError:
            console.print("[red]gdown not installed. pip install gdown[/red]")
            raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        task = progress.add_task(f"Downloading {output_path.name}", total=total)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    console.print(f"[green]OK[/green] {output_path}")


def _extract_zip(zip_path: Path, extract_to: Path, subdir: Optional[str] = None) -> None:
    """Extract zip file, optionally moving a subdirectory up."""
    console.print(f"[cyan]Extracting {zip_path.name}...[/cyan]")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    if subdir:
        subdir_path = extract_to / subdir
        if subdir_path.exists():
            for item in subdir_path.iterdir():
                dest = extract_to / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(extract_to))
            top_level = extract_to / subdir.split("/")[0]
            if top_level.exists():
                shutil.rmtree(top_level)

    console.print(f"[green]OK[/green] Extracted to {extract_to}")


def _validate_ply(ply_path: Path) -> bool:
    """Quick validation of a PLY file."""
    if not ply_path.exists():
        return False
    with open(ply_path, "rb") as f:
        header = f.read(128)
    return b"ply" in header[:16]


def _find_ply_files(directory: Path) -> list[Path]:
    """Recursively find PLY files in a directory."""
    return sorted(directory.rglob("*.ply"))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def download(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download a dataset."""
    if dataset not in DATASETS:
        console.print(f"[red]Unknown dataset '{dataset}'[/red]")
        console.print(f"Available: {', '.join(DATASETS.keys())}")
        raise typer.Exit(1)

    config = DATASETS[dataset]
    target_dir = Path(config["target_dir"])
    url = config["url"]
    dl_type = config.get("download_type", "zip")

    if url is None:
        console.print(f"[red]No download URL for '{dataset}'[/red]")
        raise typer.Exit(1)

    # Check existing
    if target_dir.exists() and not force:
        ply_files = _find_ply_files(target_dir)
        if ply_files:
            console.print(f"[yellow]Already downloaded at {target_dir} ({len(ply_files)} PLY files)[/yellow]")
            console.print("[yellow]Use --force to re-download[/yellow]")
            return

    console.print(f"[bold cyan]{config['name']}[/bold cyan]")
    console.print(f"  {config['desc']}")
    console.print(f"  Size: {config.get('size_hint', 'unknown')}")
    console.print()

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        if dl_type == "ply":
            # Single PLY file download
            target_dir.mkdir(parents=True, exist_ok=True)
            ply_path = target_dir / "point_cloud.ply"
            _download_with_progress(url, ply_path)

            if _validate_ply(ply_path):
                size_mb = ply_path.stat().st_size / (1024 * 1024)
                console.print(f"[green]OK[/green] Valid PLY: {size_mb:.1f}MB")
            else:
                console.print("[red]Downloaded file is not a valid PLY[/red]")
                raise typer.Exit(1)

        elif dl_type == "zip":
            zip_path = temp_dir / f"{dataset}.zip"
            _download_with_progress(url, zip_path)

            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            _extract_zip(zip_path, target_dir, config.get("extract_subdir"))

            # Cleanup zip
            zip_path.unlink()
            console.print(f"[dim]Cleaned up {zip_path.name}[/dim]")

        # Summary
        ply_files = _find_ply_files(target_dir)
        console.print()
        console.print(f"[bold green]Download complete![/bold green]")
        console.print(f"  Location: {target_dir}")
        console.print(f"  PLY files found: {len(ply_files)}")
        for p in ply_files[:10]:
            size_mb = p.stat().st_size / (1024 * 1024)
            console.print(f"    {p.relative_to(target_dir)} ({size_mb:.1f}MB)")
        if len(ply_files) > 10:
            console.print(f"    ... and {len(ply_files) - 10} more")

        # Usage hint
        console.print()
        if ply_files:
            console.print("[cyan]Usage with GSS import pipeline:[/cyan]")
            console.print(f"  gss run --config configs/pipeline_import.yaml")
            console.print(f"  (set s00.ply_path to a PLY file above)")

    except requests.RequestException as e:
        console.print(f"[red]Download failed:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_datasets() -> None:
    """List available datasets."""
    table = Table(title="Available 3DGS Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Status", style="dim")

    for key, config in DATASETS.items():
        target_dir = Path(config["target_dir"])
        if target_dir.exists():
            ply_count = len(_find_ply_files(target_dir))
            status = f"Downloaded ({ply_count} PLY)" if ply_count else "Downloaded"
        else:
            status = "-"
        table.add_row(
            key,
            config.get("scene_type", ""),
            config.get("desc", config["name"]),
            config.get("size_hint", "?"),
            status,
        )

    console.print(table)
    console.print()
    console.print("[cyan]Download:[/cyan] python scripts/download_sample_data.py download -d <name>")


if __name__ == "__main__":
    app()
