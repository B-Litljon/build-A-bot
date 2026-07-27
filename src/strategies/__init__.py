"""
Package marker for ``src.strategies`` -- the decision layer.

A strategy takes bars and returns a Signal or None. It never talks to a broker;
an orchestrator in src/execution calls it and acts on the answer.

Deliberately empty: ``base`` and ``concrete_strategies`` are imported by path.

Glossary:
    (none -- package marker, no identifiers of its own)
"""
