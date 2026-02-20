"""CLI entry point for the GSS pipeline.

Usage:
    gss run                     # Run full pipeline
    gss run-step s01_extract_frames  # Run single step
    gss info                    # Show pipeline info
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from gss.core.logging import setup_logging

app = typer.Typer(name="gss", help="3DGS to BIM pipeline")
console = Console()

DEFAULT_CONFIG = Path("configs/pipeline.yaml")


@app.command()
def run(config: Path = typer.Option(DEFAULT_CONFIG, help="Pipeline config path")) -> None:
    """Run the full pipeline."""
    setup_logging()
    from gss.core.pipeline_runner import run_pipeline

    run_pipeline(config)


@app.command()
def run_step(
    step_name: str = typer.Argument(..., help="Step name (e.g. s01_extract_frames)"),
    config: Path = typer.Option(DEFAULT_CONFIG, help="Pipeline config path"),
    input_json: str = typer.Option(None, "--input", "-i", help="Input as JSON string"),
) -> None:
    """Run a single pipeline step."""
    import json

    setup_logging()
    from gss.core.pipeline_runner import load_pipeline_config, import_step_class, load_step_config

    pipeline_cfg = load_pipeline_config(config)
    entry = next((s for s in pipeline_cfg.steps if s.name == step_name), None)
    if entry is None:
        console.print(f"[red]Step '{step_name}' not found in pipeline config[/red]")
        raise typer.Exit(1)

    step_cls = import_step_class(entry.module)
    step_config = load_step_config(Path(entry.config_file), step_cls.config_type)
    step_instance = step_cls(config=step_config, data_root=pipeline_cfg.data_root)

    if input_json:
        input_data = json.loads(input_json)
    else:
        schema = step_cls.input_type.model_json_schema()
        required = schema.get("required", [])
        if required:
            console.print(f"[yellow]Step '{step_name}' requires input fields: {required}[/yellow]")
            console.print("[yellow]Use --input/-i with JSON string, e.g.:[/yellow]")
            console.print(f'  gss run-step {step_name} -i \'{{"field": "value"}}\'')
            raise typer.Exit(1)
        input_data = {}

    console.print(f"[green]Running step: {step_name}[/green]")
    step_input = step_cls.input_type(**input_data)
    output = step_instance.execute(step_input)
    console.print(f"[green]Done. Output:[/green] {output.model_dump_json(indent=2)}")


@app.command()
def info(config: Path = typer.Option(DEFAULT_CONFIG, help="Pipeline config path")) -> None:
    """Show pipeline steps and their status."""
    from gss.core.pipeline_runner import load_pipeline_config

    pipeline_cfg = load_pipeline_config(config)
    table = Table(title=f"Pipeline: {pipeline_cfg.project_name}")
    table.add_column("#", style="dim")
    table.add_column("Step", style="cyan")
    table.add_column("Module", style="green")
    table.add_column("Enabled", style="yellow")
    table.add_column("Depends On", style="dim")

    for i, step in enumerate(pipeline_cfg.steps, 1):
        table.add_row(
            str(i),
            step.name,
            step.module,
            "Y" if step.enabled else "N",
            ", ".join(step.depends_on) if step.depends_on else "-",
        )
    console.print(table)


if __name__ == "__main__":
    app()
