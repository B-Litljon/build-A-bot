"""
Package marker for ``src.core`` -- shared domain types, the Discord notifier,
and the offline training / evaluation pipeline.

Deliberately empty: modules are imported by path rather than re-exported here.
Note that importers are inconsistent about the prefix -- some use
``core.retrainer``, others ``src.core.retrainer`` -- depending on whether
``src`` or the repo root is on PYTHONPATH.

Glossary:
    (none -- package marker, no identifiers of its own)
"""
