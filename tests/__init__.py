"""
Package marker for the test suite.

pyproject.toml sets ``testpaths = ["tests"]`` so a bare ``pytest`` collects only
from here. Note collection ALSO requires the default ``test_*.py`` filename
pattern -- (``verify_warmup.py``, a filename-mismatched probe, previously lived
here and was deleted with the Alpaca lane on 2026-09-16).

Glossary:
    (none -- package marker, no identifiers of its own)
"""
