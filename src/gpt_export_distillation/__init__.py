def main() -> None:
    """Run the legacy Markdown-distillation developer command lazily."""
    from .cli import main as cli_main

    cli_main()

__all__ = ["main"]
