"""
Package marker for the test suite.

pyproject.toml sets ``testpaths = ["tests"]`` so a bare ``pytest`` collects only
from here. Note collection ALSO requires the default ``test_*.py`` filename
pattern -- ``verify_warmup.py`` lives in this folder but is not collected.

Glossary:
    (none -- package marker, no identifiers of its own)
"""
