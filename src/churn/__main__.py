"""Entry point so ``python -m churn ...`` dispatches to the CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
