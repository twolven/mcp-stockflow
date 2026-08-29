#!/usr/bin/env python3
"""Compatibility launcher; stdio remains the default transport."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
if __name__ != "__main__":
    __path__ = [str(Path(__file__).parent / "src" / "stockflow")]
from stockflow.server import main

if __name__ == "__main__":
    main()
