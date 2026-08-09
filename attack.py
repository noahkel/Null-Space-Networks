#!/usr/bin/env python3
"""Adversarial attack suite — CLI entry point.

The implementation lives in ``src/attack.py`` (attack algorithms, evaluation and
suite orchestration); this file exists only so the suite can be launched as

    python attack.py --data-root ... [--lipschitz] ...
    python attack.py --epoch-study ...
    
Attacks and figures are strictly separated: this command never plots. After a
run, render (or re-render) every figure from the saved artifacts with

    python visualise.py <attacks_root>
"""
from src.attack import main

if __name__ == "__main__":
    main()
