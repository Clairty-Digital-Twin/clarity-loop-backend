"""Clarity Models CLI Tool.

Command-line interface for ML model management, deployment, and monitoring.
"""

import asyncio
import builtins
import json
from pathlib import Path
import time
from typing import Any

import aiohttp
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from clarity.ml.models.local_server import LocalModelServer, ModelServerConfig
from clarity.ml.models.registry import (
    ModelMetadata,
    ModelRegistry,
    ModelRegistryConfig,
    ModelTier,
    initialize_legacy_models,
)

console = Console()

# HTTP Status codes
HTTP_OK = 200


@click.group()
@click.option("--config-file", type=click.Path(), help="Configuration file path")
@click.option(
    "--models-dir", type=click.Path(), default="./models", help="Models directory"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(
    ctx: click.Context,
    config_file: str | None,
    models_dir: str,
    verbose: bool,  # noqa: FBT001
) -> None:
    """Clarity ML Model Management CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file
    ctx.obj["models_dir"] = Path(models_dir)
    ctx.obj["verbose"] = verbose

    if verbose:
        import logging  # noqa: PLC0415

        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.pass_context
async def init(ctx: click.Context) -> None:
    """Initialize model registry."""
    models_dir = ctx.obj["models_dir"]

    with console.status("[bold green]Initializing model registry..."):
        # Create models directory
        models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize registry
        config = ModelRegistryConfig(base_path=models_dir)
        registry = ModelRegistry(config)
        await registry.initialize()

        # Add legacy models
        await initialize_legacy_models(registry)

    console.print(f"[green]✓[/green] Model registry initialized in {models_dir}")
    console.print("[blue]Info:[/blue] Added legacy PAT models (PAT-S, PAT-M, PAT-L)")


@cli.command(name="list")
@click.pass_context
async def list_models(ctx: click.Context) -> None:
    """List all models in registry."""
    models_dir = ctx.obj["models_dir"]

    config = ModelRegistryConfig(base_path=models_dir)
    registry = ModelRegistry(config)
    await registry.initialize()

    models = await registry.list_models()

    if not models:
        console.print("[yellow]No models found in registry[/yellow]")
        return

    # Group models by model_id
    grouped_models: dict[str, builtins.list[ModelMetadata]] = {}
    for model in models:
        if model.model_id not in grouped_models:
            grouped_models[model.model_id] = []
        grouped_models[model.model_id].append(model)

    # Create table
    table = Table(title="Model Registry")
    table.add_column("Model ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Version", style="green")
    table.add_column("Tier", style="blue")
    table.add_column("Size", style="yellow")
    table.add_column("Status", style="red")

    for model_id, model_versions in grouped_models.items():
        for i, model in enumerate(sorted(model_versions, key=lambda x: x.version)):
            size_mb = model.size_bytes / (1024 * 1024)
            status = (
                "Available"
                if model.local_path and Path(model.local_path).exists()
                else "Not Downloaded"
            )

            table.add_row(
                model_id if i == 0 else "",
                model.name,
                model.version,
                model.tier.value,
                f"{size_mb:.1f} MB",
                status,
            )

    console.print(table)


@cli.command()
@click.argument("model_id")
@click.option("--version", default="latest", help="Model version")
@click.option("--source-url", help="Source URL for download")
@click.pass_context
async def download(
    ctx: click.Context, model_id: str, version: str, source_url: str | None
) -> None:
    """Download a model."""
    models_dir = ctx.obj["models_dir"]

    config = ModelRegistryConfig(base_path=models_dir)
    registry = ModelRegistry(config)
    await registry.initialize()

    # Check if model exists
    model = await registry.get_model(model_id, version)
    if not model:
        console.print(
            f"[red]Error:[/red] Model {model_id}:{version} not found in registry"
        )
        return

    console.print(f"[blue]Info:[/blue] Downloading {model.unique_id}...")

    # Start download with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:

        task = progress.add_task(f"Downloading {model.name}", total=model.size_bytes)

        # Start download
        download_task = asyncio.create_task(
            registry.download_model(model_id, version, source_url)
        )

        # Monitor progress
        while not download_task.done():
            progress_info = await registry.get_download_progress(model_id, version)
            if progress_info:
                progress.update(task, completed=progress_info["downloaded_bytes"])
            await asyncio.sleep(0.5)

        success = await download_task

        if success:
            progress.update(task, completed=model.size_bytes)
            console.print(f"[green]✓[/green] Successfully downloaded {model.unique_id}")
        else:
            console.print(f"[red]✗[/red] Failed to download {model.unique_id}")


@cli.command()
@click.argument("model_id")
@click.option("--name", required=True, help="Model name")
@click.option("--version", required=True, help="Model version")
@click.option(
    "--tier",
    type=click.Choice(["small", "medium", "large", "quantized"]),
    required=True,
    help="Model tier",
)
@click.option("--source-url", required=True, help="Source URL")
@click.option("--checksum", required=True, help="SHA-256 checksum")
@click.option("--size", type=int, required=True, help="Size in bytes")
@click.option("--description", help="Model description")
@click.option("--tags", help="Comma-separated tags")
@click.pass_context
async def register(
    ctx: click.Context,
    model_id: str,
    name: str,
    version: str,
    tier: str,
    source_url: str,
    checksum: str,
    size: int,
    description: str | None,
    tags: str | None,
) -> None:
    """Register a new model."""
    models_dir = ctx.obj["models_dir"]

    config = ModelRegistryConfig(base_path=models_dir)
    registry = ModelRegistry(config)
    await registry.initialize()

    # Create metadata
    metadata = ModelMetadata(
        model_id=model_id,
        name=name,
        version=version,
        tier=ModelTier(tier),
        size_bytes=size,
        checksum_sha256=checksum,
        source_url=source_url,
        description=description or "",
        tags=tags.split(",") if tags else [],
    )

    success = await registry.register_model(metadata)

    if success:
        console.print(f"[green]✓[/green] Registered model {metadata.unique_id}")
    else:
        console.print(f"[red]✗[/red] Failed to register model {metadata.unique_id}")


@cli.command()
@click.argument("alias")
@click.argument("model_id")
@click.argument("version")
@click.pass_context
async def alias(ctx: click.Context, alias: str, model_id: str, version: str) -> None:
    """Create model alias."""
    models_dir = ctx.obj["models_dir"]

    config = ModelRegistryConfig(base_path=models_dir)
    registry = ModelRegistry(config)
    await registry.initialize()

    success = await registry.create_alias(alias, model_id, version)

    if success:
        console.print(
            f"[green]✓[/green] Created alias '{alias}' -> {model_id}:{version}"
        )
    else:
        console.print(f"[red]✗[/red] Failed to create alias '{alias}'")


@cli.command()
@click.pass_context
async def status(ctx: click.Context) -> None:
    """Show model registry status."""
    models_dir = ctx.obj["models_dir"]

    config = ModelRegistryConfig(base_path=models_dir)
    registry = ModelRegistry(config)
    await registry.initialize()

    # Get all models and aliases
    models = await registry.list_models()
    aliases = registry.aliases

    # Calculate statistics
    total_size = sum(model.size_bytes for model in models)
    downloaded_models = [
        m for m in models if m.local_path and Path(m.local_path).exists()
    ]
    downloaded_size = sum(model.size_bytes for model in downloaded_models)

    # Create status panel
    status_text = f"""
[bold]Registry Status[/bold]

Models: {len(models)} registered
Downloaded: {len(downloaded_models)} models
Aliases: {len(aliases)} aliases
Total Size: {total_size / (1024**3):.2f} GB
Downloaded Size: {downloaded_size / (1024**3):.2f} GB
Cache Hit Rate: {(len(downloaded_models) / len(models)) * 100:.1f}%

[bold]Storage Locations[/bold]
Base Path: {config.base_path}
Cache Dir: {config.cache_dir}
Registry File: {config.registry_file}
"""

    console.print(Panel(status_text.strip(), title="Model Registry Status"))

    # Show aliases if any
    if aliases:
        alias_table = Table(title="Model Aliases")
        alias_table.add_column("Alias", style="cyan")
        alias_table.add_column("Target", style="green")
        alias_table.add_column("Updated", style="yellow")

        for alias_name, alias_obj in aliases.items():
            alias_table.add_row(
                alias_name,
                f"{alias_obj.model_id}:{alias_obj.version}",
                alias_obj.updated_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(alias_table)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")  # noqa: S104
@click.option("--port", type=int, default=8900, help="Server port")
@click.option("--auto-load/--no-auto-load", default=True, help="Auto-load models")
@click.option(
    "--create-mocks/--no-create-mocks", default=True, help="Create mock models"
)
@click.pass_context
async def serve(  # noqa: RUF029 - Click async handler
    ctx: click.Context,
    host: str,
    port: int,
    auto_load: bool,  # noqa: FBT001
    create_mocks: bool,  # noqa: FBT001
) -> None:
    """Start local model server."""
    models_dir = ctx.obj["models_dir"]

    config = ModelServerConfig(
        host=host,
        port=port,
        models_dir=models_dir,
        auto_load_models=auto_load,
        create_mock_models=create_mocks,
    )

    console.print(f"[green]Starting local model server on {host}:{port}[/green]")
    console.print(f"[blue]Models directory:[/blue] {models_dir}")
    console.print(f"[blue]Auto-load models:[/blue] {auto_load}")
    console.print(f"[blue]Create mocks:[/blue] {create_mocks}")

    server = LocalModelServer(config)

    try:
        server.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")


@cli.command()
@click.option("--url", default="http://localhost:8900", help="Server URL")
@click.argument("model_id")
@click.option("--version", default="latest", help="Model version")
@click.option("--input-file", type=click.Path(exists=True), help="Input JSON file")
@click.pass_context
async def predict(
    _ctx: click.Context, url: str, model_id: str, version: str, input_file: str | None
) -> None:
    """Make prediction using local server."""
    # Load input data
    if input_file:
        with Path(input_file).open(encoding="utf-8") as f:
            inputs = json.load(f)
    else:
        # Use sample data
        inputs = {
            "actigraphy_data": [0.1, 0.2, 0.3, 0.4, 0.5] * 100,
            "timestamps": [f"2024-01-01T{i:02d}:00:00Z" for i in range(24)],
        }

    request_data = {"model_id": model_id, "version": version, "inputs": inputs}

    console.print(f"[blue]Making prediction request to {url}[/blue]")
    console.print(f"[blue]Model:[/blue] {model_id}:{version}")

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(f"{url}/predict", json=request_data) as response,
        ):
            if response.status == HTTP_OK:
                result = await response.json()

                console.print("[green]✓[/green] Prediction successful")
                console.print(f"[blue]Latency:[/blue] {result['latency_ms']:.2f}ms")

                # Display results in a nice format
                if "outputs" in result:
                    outputs_json = json.dumps(result["outputs"], indent=2)
                    console.print(Panel(outputs_json, title="Prediction Results"))

            else:
                error = await response.text()
                console.print(f"[red]✗[/red] Prediction failed: {error}")

    except (TimeoutError, aiohttp.ClientError) as e:
        console.print(f"[red]✗[/red] Request failed: {e}")


@cli.command()
@click.option("--url", default="http://localhost:8900", help="Server URL")
@click.pass_context
async def monitor(_ctx: click.Context, url: str) -> None:
    """Monitor local server metrics."""
    console.print(f"[green]Monitoring server at {url}[/green]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")

    try:  # noqa: PLR1702 - Monitoring loop complexity
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # Get health status
                    async with session.get(f"{url}/health") as response:
                        if response.status == HTTP_OK:
                            health = await response.json()
                        else:
                            health = {"status": "error"}

                    # Get metrics
                    async with session.get(f"{url}/metrics") as response:
                        if response.status == HTTP_OK:
                            metrics = await response.json()
                        else:
                            metrics = {}

                # Clear screen and display metrics
                console.clear()

                # Health status
                status_color = "green" if health.get("status") == "healthy" else "red"
                console.print(
                    f"[{status_color}]Server Status: {health.get('status', 'unknown')}[/{status_color}]"
                )
                console.print(
                    f"[blue]Loaded Models: {health.get('loaded_models', 0)}[/blue]"
                )
                console.print(
                    f"[blue]Total Memory: {health.get('total_memory_mb', 0):.1f} MB[/blue]"
                )

                # Model metrics table
                if metrics:
                    table = Table(title="Model Performance Metrics")
                    table.add_column("Model", style="cyan")
                    table.add_column("Inferences", style="green")
                    table.add_column("Avg Latency", style="yellow")
                    table.add_column("Error Rate", style="red")
                    table.add_column("Memory", style="blue")

                    for model_id, model_metrics in metrics.items():
                        error_rate = model_metrics.get("error_count", 0) / max(
                            model_metrics.get("total_inferences", 1), 1
                        )
                        table.add_row(
                            model_id,
                            str(model_metrics.get("total_inferences", 0)),
                            f"{model_metrics.get('avg_latency_ms', 0):.1f}ms",
                            f"{error_rate:.2%}",
                            f"{model_metrics.get('memory_usage_mb', 0):.1f}MB",
                        )

                    console.print(table)

                console.print(f"\n[dim]Last updated: {time.strftime('%H:%M:%S')}[/dim]")

            except (ValueError, TypeError, KeyError, RuntimeError) as e:
                console.print(f"[red]Monitoring error: {e}[/red]")

            await asyncio.sleep(2)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")


@cli.command()
@click.option(
    "--max-size-gb", type=float, default=10.0, help="Maximum cache size in GB"
)
@click.pass_context
async def cleanup(ctx: click.Context, max_size_gb: float) -> None:
    """Clean up model cache."""
    models_dir = ctx.obj["models_dir"]

    config = ModelRegistryConfig(base_path=models_dir)
    registry = ModelRegistry(config)
    await registry.initialize()

    with console.status("[bold yellow]Cleaning up cache..."):
        removed_count = await registry.cleanup_cache(max_size_gb)

    if removed_count > 0:
        console.print(f"[green]✓[/green] Cleaned up {removed_count} cached files")
    else:
        console.print(
            f"[blue]Info:[/blue] Cache is within size limit ({max_size_gb}GB)"
        )


def main() -> None:
    """Main CLI entry point."""

    # Convert sync CLI to async
    def async_cli() -> Any:
        return asyncio.run(cli())

    # Replace click commands with async versions
    for command in cli.commands.values():
        if asyncio.iscoroutinefunction(command.callback):
            original_callback = command.callback
            command.callback = (
                lambda *args, cb=original_callback, **kwargs: asyncio.run(
                    cb(*args, **kwargs)
                )
            )

    cli()


if __name__ == "__main__":
    main()
