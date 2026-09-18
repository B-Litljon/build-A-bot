"""
`python -m src.core.retrainer` entry point (package form since 2026-09-16).

The pipeline was split out of the monolithic retrainer.py into the modules
beside this file; this shim keeps the invocation working and exits with the
pipeline's status code (0=promoted, 1=error, 2=gate rejected).
"""
import sys

from ._pipeline import main

if __name__ == "__main__":
    sys.exit(main())
