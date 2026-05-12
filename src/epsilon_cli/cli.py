"""Epsilon CLI — Click interface."""
import click
from . import __version__
from .resonance import run_optimization_benchmark


@click.command()
@click.version_option(version=__version__)
def main():
    """Epsilon CLI — Context Window Optimization via Stochastic Resonance.

    Stochastically inject noise into LLM training signal paths to escape
    local minima and improve context utilization.
    """
    click.echo(f"epsilon-cli {__version__}")
    click.echo("Stochastic Resonance: adaptive noise injection for context optimization")


@click.command()
def info():
    """Show epsilon-cli version and configuration."""
    click.echo(f"epsilon-cli {__version__}")
    click.echo("Stochastic Resonance: adaptive noise injection for context optimization")


@click.command("stochastic")
@click.option("--steps", default=100, help="Number of optimization steps")
@click.option("--dim", default=128, help="Embedding dimension")
@click.option("--seed", default=42, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print gain schedule")
def stochastic(steps, dim, seed, verbose):
    """Run stochastic resonance optimization benchmark."""
    click.echo(f"Running SR benchmark: dim={dim}, steps={steps}, seed={seed}")

    result = run_optimization_benchmark(dim=dim, steps=steps, seed=seed)

    click.echo(f"Final loss: {result['final_loss']:.6f}")
    click.echo(f"Initial gain: {result['gain_schedule'][0]:.6f}")
    click.echo(f"Final gain:   {result['gain_schedule'][-1]:.6f}")

    if verbose:
        click.echo("\nGain schedule (every 10 steps):")
        for i in range(0, len(result['gain_schedule']), 10):
            click.echo(f"  step {i:4d}: gain={result['gain_schedule'][i]:.6f}  loss={result['loss_curve'][i]:.6f}")


if __name__ == "__main__":
    main()
