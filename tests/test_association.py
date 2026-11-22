import numpy as np
import pandas as pd
import pytest
from scipy import stats

import triger


def simulate_layers(N=100, m=40, n=30):
    rng = np.random.default_rng(5)
    f = rng.standard_normal(N)
    X = rng.standard_normal((N, m))
    Y = rng.standard_normal((N, n))
    X[:, :12] += f[:, None]
    Y[:, :8] += f[:, None]
    samples = [f"s{i}" for i in range(N)]
    X = pd.DataFrame(X, index=samples, columns=[f"g{i}" for i in range(m)])
    Y = pd.DataFrame(Y, index=samples, columns=[f"m{i}" for i in range(n)])
    return X, Y


def test_association_matches_pearsonr():
    X, Y = simulate_layers()
    a = triger.association_matrices(X, Y)
    assert a.R.shape == (70, 70) and a.m == 40 and a.n == 30
    assert a.names[0] == "g0" and a.names[40] == "m0"
    assert np.all(np.diag(a.P) == 0)
    r, p = stats.pearsonr(X["g0"], Y["m0"])
    assert a.R[0, 40] == pytest.approx(r)
    assert a.P[0, 40] == pytest.approx(-np.log10(p))
    s = triger.association_matrices(X, Y, method="spearman")
    rho, _ = stats.spearmanr(X["g1"], Y["m2"])
    assert s.R[1, 42] == pytest.approx(rho)


def test_association_names_for_arrays():
    X, Y = simulate_layers()
    a = triger.association_matrices(X.to_numpy(), Y.to_numpy())
    assert a.names[:2] == ["X0", "X1"] and a.names[40] == "Y0"


def test_association_rejects_invalid_input():
    X, Y = simulate_layers()
    with pytest.raises(ValueError, match="same number of samples"):
        triger.association_matrices(X, Y.iloc[1:])
    Xc = X.copy()
    Xc["g3"] = 1.0
    with pytest.raises(ValueError, match="zero variance: g3"):
        triger.association_matrices(Xc, Y)
    Xc.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        triger.association_matrices(Xc, Y)
    with pytest.raises(ValueError, match="method"):
        triger.association_matrices(X, Y, method="kendall")


def test_triger_recovers_module_with_names():
    X, Y = simulate_layers()
    res = triger.triger(X, Y, lambda1=1.3, X_size=(3, 50), Y_size=(3, 50))
    assert isinstance(res, triger.TrigerResult)
    mod = res.modules[0]
    assert mod._fields == ("X", "Y", "density", "X_names", "Y_names")
    assert len(set(mod.X_names) & {f"g{i}" for i in range(12)}) >= 10
    assert len(set(mod.Y_names) & {f"m{i}" for i in range(8)}) >= 6
    assert mod.X_names == [X.columns[i] for i in mod.X]


def test_triger_matches_samples_by_index():
    X, Y = simulate_layers()
    ref = triger.triger(X, Y, lambda1=1.3, X_size=(3, 50), Y_size=(3, 50))
    shuffled = triger.triger(X, Y.iloc[::-1], lambda1=1.3, X_size=(3, 50),
                             Y_size=(3, 50))
    for a, b in zip(ref.modules, shuffled.modules):
        assert a.X.tolist() == b.X.tolist() and a.Y.tolist() == b.Y.tolist()
    Yb = Y.rename(index={"s0": "other"})
    with pytest.raises(ValueError, match="sample names"):
        triger.triger(X, Yb)


def test_triger_passes_arguments_to_greedy_3step():
    X, Y = simulate_layers()
    res = triger.triger(X, Y, lambda1=1.3, X_size=(3, 50), Y_size=(3, 50),
                        max_blocks=1)
    assert len(res.modules) == 1


def test_triger_arrays_and_dataframes_agree():
    X, Y = simulate_layers()
    a = triger.triger(X, Y, lambda1=1.3, X_size=(3, 50), Y_size=(3, 50))
    b = triger.triger(X.to_numpy(), Y.to_numpy(), lambda1=1.3,
                      X_size=(3, 50), Y_size=(3, 50))
    assert [m.X.tolist() for m in a.modules] == \
        [m.X.tolist() for m in b.modules]
