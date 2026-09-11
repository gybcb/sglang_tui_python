from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="sglang-tui", description="btop-style dashboard for SGLang servers")
    parser.add_argument("--url", default="http://localhost:30000", help="SGLang server base URL (default: %(default)s)")
    parser.add_argument("--refresh", type=float, default=1.0, help="poll interval in seconds (default: %(default)s)")
    args = parser.parse_args()

    from .app import SglangTui

    SglangTui(url=args.url, refresh=args.refresh).run()


if __name__ == "__main__":
    main()
