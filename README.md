# triger

Python package for **TriGer**, from the paper *Structured cross-omics interaction
discovery with a triple-graph model*.

TriGer finds modules of features that are densely associated across two omics
layers (X and Y) and coherent within each layer. It peels dense blocks from the
cross-omics association matrix, then refines large blocks using the
within-omics correlation matrices.

## Installation

```bash
pip install git+https://github.com/jy2941/TriGer-PyPI.git
```

## Quick start

```python
import numpy as np
from scipy import stats          # only used here to compute p-values
import triger

# X: samples x m features, Y: samples x n features
N, m, n = X.shape[0], X.shape[1], Y.shape[1]
R = np.corrcoef(np.hstack([X, Y]).T)                 # signed correlations
r = np.clip(R, -0.999999, 0.999999)
t = r * np.sqrt((N - 2) / (1 - r ** 2))
P = -np.log10(2 * stats.t.sf(np.abs(t), N - 2))      # -log10 p
np.fill_diagonal(P, 0)

blocks = triger.greedy_3step(P, R, m, n, lambda1=1.1, lambda2=1.8, lambda3=1.8,
                             X_size=(10, 15), Y_size=(10, 250))
for x_idx, y_idx in blocks:
    print(len(x_idx), "X features,", len(y_idx), "Y features")

triger.plot3in1_v1(R[:m, :m], R[m:, m:], R[:m, m:], blocks, line=True)
```

## Functions

| Function | Description |
|----------|-------------|
| `greedy_3step` | Full pipeline: cross-omics block detection, then within-omics refinement of large blocks |
| `greedy_peeling_XY_one` / `greedy_peeling_XY_all` | Dense blocks in a cross-omics (X-Y) matrix |
| `greedy_peeling_X_one` / `greedy_peeling_X_all` | Dense communities in a within-omics (symmetric) matrix |
| `sort_result` / `sort_result_XandY` | Order features within each block |
| `plot3in1_v1` | X-Y heatmap with the X-X and Y-Y matrices as triangles |
| `plot_all_result` | Heatmap of the detected blocks |

All indices are 0-based. `lambda1` is the cross-omics size penalty and `lambda2` /
`lambda3` are the within-omics penalties for X and Y (λ₀, λ₁, λ₂ in the paper);
higher values give smaller, denser blocks.

## Related

- R package: [TriGer-R](https://github.com/jy2941/TriGer-R)
- Code for the paper's simulations and analyses: [TriGer](https://github.com/jy2941/TriGer)

## License

MIT
