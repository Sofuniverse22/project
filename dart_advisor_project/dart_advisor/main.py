"""Main entry point"""
import click
from pathlib import Path
from rich.console import Console

console = Console()

@click.group()
def cli():
    """DART Advisor - AI-Powered Investment Analysis Platform"""
    pass

@cli.command()
def init():
    """Initialize DART Advisor"""
    console.print("🚀 Initializing DART Advisor...")

    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            console.print("✓ Created .env", style="green")
            console.print("⚠️  Please edit .env and add your ANTHROPIC_API_KEY", style="yellow")

    for dir_name in ["output", "logs", ".cache"]:
        Path(dir_name).mkdir(exist_ok=True)
        console.print(f"✓ Created {dir_name}/", style="green")

    console.print("\n✅ Initialization complete!", style="bold green")

@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option('--company', '-c', required=True, help='Company name')
@click.option('--output', '-o', help='Output PDF path')
def analyze(files, company, output):
    """Analyze company documents"""
    console.print(f"🔍 Analyzing {company}...", style="bold blue")
    console.print(f"📄 Files: {len(files)}", style="cyan")
    console.print("\n⚠️  Implementation in progress...", style="yellow")

if __name__ == "__main__":
    cli()
