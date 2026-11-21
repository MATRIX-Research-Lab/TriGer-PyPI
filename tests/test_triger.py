import numpy as np
import matplotlib
import pytest

matplotlib.use("Agg")
import triger  # noqa: E402


def planted(seed=0, m=60, n=80):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 0.5, (m, n))
    W[5:20, 10:30] += 4
    return W


def test_xy_one_recovers_block():
    _, x, y = triger.greedy_peeling_XY_one(planted(), 1.5)
    assert set(range(5, 20)) <= set(x.tolist())
    assert set(range(10, 30)) <= set(y.tolist())


def test_xy_all_first_block():
    blocks = triger.greedy_peeling_XY_all(planted(), 1.5)
    x, y, density = blocks[0]
    assert set(x.tolist()) == set(range(5, 20))
    assert set(y.tolist()) == set(range(10, 30))
    assert density > 0
    assert isinstance(blocks[0], triger.Block)
    assert blocks[0].X is x


def test_max_blocks_is_maximum_number_of_blocks():
    rng = np.random.default_rng(3)
    W = rng.uniform(0, 0.5, (100, 100))
    for k in range(6):
        r = slice(k * 15, k * 15 + 10)
        W[r, r] += 4
    assert len(triger.greedy_peeling_XY_all(W, 1.5, max_blocks=6)) == 6
    assert len(triger.greedy_peeling_XY_all(W, 1.5, max_blocks=2)) == 2
    assert len(triger.greedy_peeling_XY_all(W, 1.5)) == 5
    with pytest.raises(ValueError, match="max_blocks"):
        triger.greedy_peeling_XY_all(W, 1.5, max_blocks=0)


def test_x_all_symmetric():
    rng = np.random.default_rng(1)
    S = rng.uniform(-0.2, 0.2, (50, 50))
    S[:12, :12] += 0.9
    S = (S + S.T) / 2
    comms = triger.greedy_peeling_X_all(S, 1.5)
    assert set(range(12)) <= set(comms[0].X.tolist())
    with pytest.raises(ValueError, match="symmetric"):
        triger.greedy_peeling_X_all(np.arange(4.0).reshape(2, 2), 1.5)
    with pytest.raises(ValueError, match="symmetric"):
        triger.greedy_peeling_X_one(np.arange(4.0).reshape(2, 2), 1.5)


def test_min_size_drops_small_communities():
    S = np.eye(6)
    S[:3, :3] = 0.8
    np.fill_diagonal(S, 1)
    all_comms = triger.greedy_peeling_X_all(S, 1.9)
    kept = triger.greedy_peeling_X_all(S, 1.9, min_size=2)
    assert any(len(c.X) == 1 for c in all_comms)
    assert all(len(c.X) >= 2 for c in kept)


def test_sort_functions_keep_members():
    W = planted()
    blocks = triger.greedy_peeling_XY_all(W, 1.5)
    s = triger.sort_result(W, blocks)
    assert set(s[0].X.tolist()) == set(blocks[0].X.tolist())
    assert isinstance(s[0], triger.Block)
    X = np.corrcoef(np.random.default_rng(2).standard_normal((30, 60)).T)
    Y = np.corrcoef(np.random.default_rng(3).standard_normal((30, 80)).T)
    s2 = triger.sort_result_XandY(X, Y, blocks)
    assert set(s2[0].Y.tolist()) == set(blocks[0].Y.tolist())


def test_greedy_3step():
    rng = np.random.default_rng(4)
    N, m, n = 150, 80, 120
    F = rng.standard_normal(N)
    X = rng.standard_normal((N, m))
    Y = rng.standard_normal((N, n))
    X[:, :25] += 1.5 * F[:, None]
    Y[:, :40] += 1.5 * F[:, None]
    R = np.corrcoef(np.hstack([X, Y]).T)
    P = np.abs(R)
    np.fill_diagonal(P, 0)
    blocks = triger.greedy_3step(P, R, m, n, 1.1, 1.8, 1.8,
                                 X_size=(10, 200), Y_size=(10, 200))
    assert blocks
    assert len(set(blocks[0].X.tolist()) & set(range(25))) >= 20
    assert blocks[0]._fields == ("X", "Y", "density")
    assert blocks[0].Y.max() < n


def test_invalid_input_is_rejected():
    W = planted()
    W[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        triger.greedy_peeling_XY_one(W, 1.5)
    with pytest.raises(ValueError, match="finite number"):
        triger.greedy_peeling_XY_one(planted(), "a")
    with pytest.raises(ValueError, match="numeric"):
        triger.greedy_peeling_XY_all([["a", "b"], ["c", "d"]])
    P = np.zeros((5, 5))
    with pytest.raises(ValueError, match=r"\(m \+ n\)"):
        triger.greedy_3step(P, P, 2, 2)
    with pytest.raises(ValueError, match="X_size"):
        triger.greedy_3step(P, P, 2, 3, X_size=(5, 1))


def test_plots_run():
    W = planted()
    blocks = triger.sort_result(W, triger.greedy_peeling_XY_all(W, 1.5))
    X = np.corrcoef(np.random.default_rng(5).standard_normal((40, 60)).T)
    Y = np.corrcoef(np.random.default_rng(6).standard_normal((40, 80)).T)
    triger.plot_all_result(W, blocks, full=True)
    triger.plot3in1_v1(X, Y, W, blocks, line=True)
