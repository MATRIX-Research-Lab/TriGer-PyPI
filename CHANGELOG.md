# Changelog

## 1.0.0

First release on PyPI.

- `triger()` runs the full pipeline on two omics layers given as NumPy arrays
  or pandas DataFrames, matches DataFrame samples by index, and returns
  feature names with each module.
- `association_matrices()` computes the signed correlation and -log10 p-value
  matrices used by `greedy_3step()`.
- Results are `Block`, `Community`, `Module` and `TrigerResult` records
  (named tuples).
- `greedy_peeling_XY_all()` and `greedy_3step()` take `max_blocks`, the
  maximum number of blocks returned (default 5). `greedy_3step()` returns the
  block density.
- Invalid input raises `ValueError`.
- Tests compare the peeling functions with reference outputs shared with the
  R package.
