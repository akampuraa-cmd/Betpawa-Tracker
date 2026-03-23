"""
Entry point for the Betpawa Tracker application.

Usage
-----
  python main.py              → launch the Tkinter GUI
  python main.py --cli <cmd>  → run a CLI command (see `python cli.py --help`)
  python main.py --help       → show help

Examples
--------
  python main.py                          # GUI
  python main.py --cli start              # CLI scheduler
  python main.py --cli scrape             # CLI one-shot scrape
  python main.py --cli results --count 10 # CLI show results
  python main.py --cli train              # CLI train AI
  python main.py --cli predict            # CLI predict
  python main.py --cli status             # CLI status
"""

import sys
import argparse
import logging


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    # Peek at sys.argv to decide whether to launch GUI or CLI
    parser = argparse.ArgumentParser(
        prog="betpawa-tracker",
        description=(
            "Betpawa MUN Tracker — web scraper + AI\n\n"
            "Run without arguments to launch the GUI.\n"
            "Pass --cli <command> to use the command-line interface."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cli",
        nargs=argparse.REMAINDER,
        metavar="COMMAND [ARGS]",
        help=(
            "Run a CLI command instead of the GUI. "
            "Supported commands: start, scrape, results, train, predict, status"
        ),
    )

    args = parser.parse_args()

    _setup_logging()

    if args.cli is not None:
        # Delegate to the CLI module
        import cli as cli_module
        return cli_module.main(args.cli)
    else:
        # Launch GUI
        try:
            import tkinter as tk  # noqa: F401 – just to check availability
        except ImportError:
            print(
                "Tkinter is not available in this Python installation.\n"
                "Install it (e.g., `sudo apt install python3-tk`) or use the "
                "CLI mode: python main.py --cli --help"
            )
            return 1

        import gui as gui_module
        gui_module.launch()
        return 0


if __name__ == "__main__":
    sys.exit(main())
