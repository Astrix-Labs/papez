"""Allow running with: python -m papez or papez CLI entry point."""
from __future__ import annotations

import asyncio

from papez.server import main as server_main


def main() -> None:
    """Entry point for console script."""
    asyncio.run(server_main())


if __name__ == "__main__":
    main()
