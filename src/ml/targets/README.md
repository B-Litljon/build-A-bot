# `src/ml/targets`

Target generators — the "right answer" column a model is trained against.

⚠️ **The production labels are not here.** `src/core/retrainer.py` builds its
own labels inline. This folder holds the older, simpler definition.

No `__init__.py` — implicit namespace package.

## Files

### `v3_targets.py` — ⚠️ legacy
`V3DirectionalTarget` labels a bar `1` if price rises by at least
`_MIN_GAIN_PCT` (0.3%) within the next `_LOOKAHEAD_BARS` (15) bars, else `0`.
Rows at the end, where the future isn't known, are labelled **null rather than
0** so they get dropped instead of being mislabelled as losses.

The important contrast with production: this only asks whether price *ever
reached* a level. It ignores whether a stop would have been hit on the way
there. The retrainer's labels replay stops and targets bar by bar and check the
stop first, which is why they are harsher and much closer to what actually
happens to a live order.

- **Imports from repo:** `ml.core.interfaces`.
- **Imported by:** `src/ml/feature_pipeline.py` only (used as the default
  target generator in its `main()` script path).
- **Data artifacts:** none.
