#!/usr/bin/env python3
"""Attack-suite figure rendering — CLI entry point.

The implementation lives in ``src/visualisations.py`` (every matplotlib figure
plus the .npz/.json artifact read/write helpers). This thin wrapper re-exports
it and drives ``render_tree`` from the command line:

    python visualise.py attacks_n0.01

It rebuilds every figure of a saved run purely from the artifacts that
``attack.py`` wrote — no models, no radon operator required — so plots
can be regenerated (e.g. after tweaking a figure) without re-attacking.
"""
import argparse

from src.visualisations import render_tree


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render attack-suite figures from saved artifacts (no torch, "
                    "models or radon operator required).")
    parser.add_argument("attacks_root",
                        help="attacks_n<noise> directory (or a single init_<init> "
                             "directory) produced by attack.py.")
    args = parser.parse_args()
    render_tree(args.attacks_root)


if __name__ == "__main__":
    main()
