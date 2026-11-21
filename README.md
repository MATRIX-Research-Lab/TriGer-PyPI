# triger

Python package for **TriGer**, from the paper *Structured cross-omics interaction
discovery with a triple-graph model*.

TriGer finds modules of features that are densely associated across two omics
layers (X and Y) and coherent within each layer. It peels dense blocks from the
cross-omics association matrix, then refines large blocks using the
within-omics correlation matrices.

## Installation

```bash
pip install triger
```

The development version:

```bash
pip install git+https://github.com/MATRIX-Research-Lab/TriGer-PyPI.git
```

## Quick start

```python
import triger

# X: samples x m features, Y: samples x n features (NumPy arrays or pandas
# DataFrames; DataFrames are matched by sample index and keep feature names)
res = triger.triger(X, Y, lambda1=1.1, lambda2=1.8, lambda3=1.8,
                    X_size=(10, 15), Y_size=(10, 250))

for mod in res.modules:
    print(len(mod.X), "X features,", len(mod.Y), "Y features")
print(res.modules[0].X_names)

# Cross- and within-omics correlations of the module features
m = res.m
triger.plot3in1_v1(res.R[:m, :m], res.R[m:, m:], res.R[:m, m:], res.modules,
                   line=True)
```

The lower-level functions work on precomputed matrices:

```python
a = triger.association_matrices(X, Y)          # signed R and -log10 p P
blocks = triger.greedy_3step(a.P, a.R, a.m, a.n, lambda1=1.1, lambda2=1.8,
                             lambda3=1.8, X_size=(10, 15), Y_size=(10, 250))
x_idx, y_idx, density = blocks[0]              # or blocks[0].X, .Y, .density
```

## Functions

| Function | Description |
|----------|-------------|
| `triger` | Full pipeline on two omics layers |
| `association_matrices` | Signed correlation and -log10 p-value matrices over both layers |
| `greedy_3step` | Full pipeline on precomputed association matrices |
| `greedy_peeling_XY_one` / `greedy_peeling_XY_all` | Dense blocks in a cross-omics (X-Y) matrix |
| `greedy_peeling_X_one` / `greedy_peeling_X_all` | Dense communities in a within-omics (symmetric) matrix |
| `sort_result` / `sort_result_XandY` | Order features within each block |
| `plot3in1_v1` | X-Y heatmap with the X-X and Y-Y matrices as triangles |
| `plot_all_result` | Heatmap of the detected blocks |

Blocks are returned as `Block(X, Y, density)` records, communities as
`Community(X, density)`, and the modules of `triger` as
`Module(X, Y, density, X_names, Y_names)`. All indices are 0-based.

`lambda1` is the cross-omics size penalty and `lambda2` / `lambda3` are the
within-omics penalties for X and Y (λ₀, λ₁, λ₂ in the paper); higher values give
smaller, denser blocks. `max_blocks` is the maximum number of blocks found in
the cross-omics step (default 5). The CRC and IBD analyses of the paper, run
with the MATLAB code, correspond to `max_blocks=4`.

The R package gives identical results on the same input; both packages are
tested against shared reference outputs.

## Related

- R package: [TriGer](https://github.com/MATRIX-Research-Lab/TriGer)
- Code for the paper's simulations and analyses: [TriGer](https://github.com/jy2941/TriGer)

## License

MIT
