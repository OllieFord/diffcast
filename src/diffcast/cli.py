"""DiffCast command-line interface."""

import typer

app = typer.Typer(
    name="diffcast",
    help="DiffCast: Diffusion-based day-ahead electricity price forecasting",
)


@app.command()
def version() -> None:
    """Show version."""
    from diffcast import __version__

    typer.echo(f"DiffCast v{__version__}")


@app.command()
def download(
    api_key: str = typer.Option(
        ...,
        "--api-key",
        "-k",
        envvar="ENTSOE_API_KEY",
        help="ENTSO-E API key",
    ),
    output_dir: str = typer.Option(
        "data/raw",
        "--output",
        "-o",
        help="Output directory",
    ),
) -> None:
    """Download raw data from ENTSO-E and Open-Meteo."""
    import subprocess

    subprocess.run(
        ["python", "scripts/download_data.py", "--api-key", api_key, "--output", output_dir],
        check=True,
    )


@app.command()
def prepare() -> None:
    """Process raw data and create train/val/test splits."""
    import subprocess

    subprocess.run(["python", "scripts/prepare_dataset.py"], check=True)


@app.command()
def train(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Config file path",
    ),
    epochs: int = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Max epochs",
    ),
) -> None:
    """Train DiffCast model."""
    import subprocess

    cmd = ["python", "scripts/train.py", "--config", config]
    if epochs:
        cmd.extend(["--max-epochs", str(epochs)])
    subprocess.run(cmd, check=True)


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(..., help="Model checkpoint path"),
    output: str = typer.Option(
        "results",
        "--output",
        "-o",
        help="Output directory",
    ),
) -> None:
    """Evaluate trained model."""
    import subprocess

    subprocess.run(
        ["python", "scripts/evaluate.py", checkpoint, "--output", output],
        check=True,
    )


if __name__ == "__main__":
    app()
