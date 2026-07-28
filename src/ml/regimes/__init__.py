"""
Market regime detection modules (HMM, change-point, etc.).

Only ``hmm_regime`` exists today, and it is off unless RETRAIN_USE_HMM=1.
"Regime" here means a hidden market mode (quiet, trending, violent) that nobody
labels directly -- it is inferred statistically and handed to the classifier as
extra inputs. See GLOSSARY.md.

Glossary:
    (none -- package marker, no identifiers of its own)
"""
