"""
Fallback and fixed-mode value for the Angel proposal threshold.

Before 2026-07-27 this value was hardcoded independently in seven files, so a
threshold sweep meant editing all of them in lockstep — or silently
desynchronising the stages. The coupling is real, not stylistic: the Devil
trains ONLY on Angel-approved rows (retrainer Phase 5.5) and the bracket
optimizer fits stop/target distances on that same population, so the live
gate, the retrainer, and every analysis tool must agree on one value or the
model pair quietly drifts apart from the population it was fitted on
(see GLOSSARY.md: Angel, Devil, threshold).

Since 2026-08-29 this constant is the DEFAULT, not the production value: the
retrainer calibrates the bar per model from out-of-fold score distributions
(``_find_optimal_angel_threshold`` in core/retrainer.py) because a fixed bar
cannot track score distributions that shift with model capacity — a 0.40
bar against the compressed scores of a 15-leaf model passed ~1 proposal per
30-100k bars (gate matrix, 2026-08-29). Setting the ANGEL_THRESHOLD env var
explicitly selects the old fixed-bar mode.

Precedence on the live side: MLStrategy treats this constant as a DEFAULT and
overrides it with the ``angel_threshold`` pinned in the model directory's
``threshold.json`` when present, so a deployed model always runs at the bar
its Devil and brackets were fitted for, regardless of the environment it
boots in. The env override is therefore a train-time and analysis-time knob,
not a live-tuning knob.

Glossary:
    ANGEL_THRESHOLD -- probability at/above which stage one (the Angel)
        proposes a trade. 0.40 unless the ANGEL_THRESHOLD env var overrides
        it at process start. Fallback + fixed-mode value; the retrainer's
        OOF calibration supersedes it per artifact. Imported by the retrainer
        (Devil training population + validation gate), both legacy trainers,
        the bracket optimizer, the failure-mode analyzer, and both live
        orchestrators.
"""

import os

ANGEL_THRESHOLD: float = float(os.getenv("ANGEL_THRESHOLD", "0.40"))
