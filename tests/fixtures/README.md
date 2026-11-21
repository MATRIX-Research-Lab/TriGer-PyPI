# Reference fixtures

Copies of the fixtures in `tests/testthat/fixtures/` of the R package
([TriGer](https://github.com/MATRIX-Research-Lab/TriGer)), where
`inst/scripts/make_fixtures.R` describes each file and the function settings
used, and `inst/scripts/make_fixtures.m` regenerates the expected outputs with
the original MATLAB implementation. The current expected outputs were generated
by the R implementation and agree with this package.

Block indices in the CSV files are 1-based; the tests subtract 1. Densities are
compared with a relative tolerance of 1e-6, because the MATLAB implementation
computes the peeling in single precision.
