"""Download sample datasets for GSS pipeline testing.

Usage:
    python scripts/download_sample_data.py --dataset lego
    python scripts/download_sample_data.py --dataset room_sample
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

app = typer.Typer(name="download_sample_data", help="Download sample datasets for GSS pipeline")
console = Console()
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "lego": {
        "name": "NeRF Synthetic Lego",
        "url": "https://huggingface.co/datasets/dylanebert/NeRF-Synthetic/resolve/main/nerf_synthetic.zip",
        "target_dir": "data/raw/lego",
        "extract_subdir": "lego",  # The folder name inside the zip
        "expected_files": ["transforms_train.json", "transforms_val.json", "transforms_test.json"],
        "expected_dirs": ["train", "val", "test"],
    },
    "room_sample": {
        "name": "Room Sample (placeholder)",
        "url": None,
        "target_dir": "data/raw/room_sample",
        "extract_subdir": None,
        "expected_files": [],
        "expected_dirs": [],
    },
}


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        task = progress.add_task(f"Downloading to {output_path.name}", total=total_size)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    console.print(f"[green]✓[/green] Downloaded to {output_path}")


def extract_zip(zip_path: Path, extract_to: Path, subdir: Optional[str] = None) -> None:
    """Extract a zip file with progress."""
    console.print(f"[cyan]Extracting {zip_path.name}...[/cyan]")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # If there's a subdirectory, move its contents up
    if subdir:
        subdir_path = extract_to / subdir
        if subdir_path.exists():
            # Move all files from subdir to extract_to
            for item in subdir_path.iterdir():
                dest = extract_to / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(extract_to))
            # Remove the now-empty subdirectory
            subdir_path.rmdir()

    console.print(f"[green]✓[/green] Extracted to {extract_to}")


def validate_dataset(target_dir: Path, config: dict) -> bool:
    """Validate that the dataset has expected structure."""
    console.print(f"[cyan]Validating dataset structure...[/cyan]")

    if not target_dir.exists():
        console.print(f"[red]✗[/red] Target directory does not exist: {target_dir}")
        return False

    # Check expected files
    for expected_file in config["expected_files"]:
        file_path = target_dir / expected_file
        if not file_path.exists():
            console.print(f"[red]✗[/red] Missing expected file: {expected_file}")
            return False

        # Validate JSON files
        if file_path.suffix == ".json":
            try:
                data = json.loads(file_path.read_text())
                if "frames" not in data:
                    console.print(f"[red]✗[/red] Invalid JSON structure in {expected_file}: missing 'frames' key")
                    return False
                console.print(f"[green]✓[/green] Valid JSON: {expected_file} ({len(data['frames'])} frames)")
            except json.JSONDecodeError as e:
                console.print(f"[red]✗[/red] Invalid JSON in {expected_file}: {e}")
                return False

    # Check expected directories
    for expected_dir in config["expected_dirs"]:
        dir_path = target_dir / expected_dir
        if not dir_path.exists():
            console.print(f"[red]✗[/red] Missing expected directory: {expected_dir}")
            return False

        # Count images in directory
        image_extensions = {".png", ".jpg", ".jpeg"}
        images = [f for f in dir_path.iterdir() if f.suffix.lower() in image_extensions]
        console.print(f"[green]✓[/green] Found directory: {expected_dir} ({len(images)} images)")

    console.print(f"[green]✓[/green] Dataset validation passed")
    return True


@app.command()
def download(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name (lego, room_sample)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if exists"),
) -> None:
    """Download and validate sample datasets."""
    if dataset not in DATASETS:
        console.print(f"[red]Error:[/red] Unknown dataset '{dataset}'")
        console.print(f"[yellow]Available datasets:[/yellow] {', '.join(DATASETS.keys())}")
        raise typer.Exit(1)

    config = DATASETS[dataset]
    target_dir = Path(config["target_dir"])

    # Check if already downloaded
    if target_dir.exists() and not force:
        console.print(f"[yellow]Dataset already exists at {target_dir}[/yellow]")
        console.print(f"[yellow]Use --force to re-download[/yellow]")
        if validate_dataset(target_dir, config):
            console.print(f"[green]✓[/green] Dataset is valid and ready to use")
            return
        else:
            console.print(f"[yellow]Validation failed, proceeding with download...[/yellow]")

    # Handle datasets without URL
    if config["url"] is None:
        console.print(f"[red]Error:[/red] Dataset '{dataset}' does not have a download URL configured")
        console.print(f"[yellow]Please manually place data in {target_dir}[/yellow]")
        raise typer.Exit(1)

    # Create temp directory for download
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        console.print(f"[cyan]Downloading {config['name']}...[/cyan]")
        zip_path = temp_dir / f"{dataset}.zip"
        download_file(config["url"], zip_path)

        # Clean up existing directory
        if target_dir.exists():
            console.print(f"[yellow]Removing existing directory...[/yellow]")
            shutil.rmtree(target_dir)

        # Extract
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        extract_zip(zip_path, target_dir, config.get("extract_subdir"))

        # Validate
        if validate_dataset(target_dir, config):
            console.print(f"[green]✓[/green] Successfully downloaded and validated {config['name']}")
            console.print(f"[green]✓[/green] Dataset ready at: {target_dir}")
        else:
            console.print(f"[red]✗[/red] Dataset validation failed")
            raise typer.Exit(1)

    except requests.RequestException as e:
        console.print(f"[red]Error downloading:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        # Cleanup temp files
        if zip_path.exists():
            zip_path.unlink()
            console.print(f"[dim]Cleaned up temporary file: {zip_path}[/dim]")


@app.command()
def list_datasets() -> None:
    """List available datasets."""
    from rich.table import Table

    table = Table(title="Available Sample Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Target Directory", style="yellow")
    table.add_column("Status", style="dim")

    for key, config in DATASETS.items():
        target_dir = Path(config["target_dir"])
        status = "✓ Downloaded" if target_dir.exists() else "Not downloaded"
        table.add_row(key, config["name"], config["target_dir"], status)

    console.print(table)


if __name__ == "__main__":
    app()
