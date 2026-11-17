import numpy as np
import matplotlib
matplotlib.use("Agg")
import triger


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
    assert set(x.tolist()) == set(range(5, 20)) and set(y.tolist()) == set(range(10, 30))
    assert density > 0


def test_x_all_symmetric():
    rng = np.random.default_rng(1)
    S = rng.uniform(-0.2, 0.2, (50, 50)); S[:12, :12] += 0.9; S = (S + S.T) / 2
    comms = triger.greedy_peeling_X_all(S, 1.5)
    assert set(range(12)) <= set(comms[0][0].tolist())


def test_sort_functions_keep_members():
    W = planted()
    blocks = triger.greedy_peeling_XY_all(W, 1.5)
    s = triger.sort_result(W, blocks)
    assert set(s[0][0].tolist()) == set(blocks[0][0].tolist())
    X = np.corrcoef(np.random.default_rng(2).standard_normal((30, 60)).T)
    Y = np.corrcoef(np.random.default_rng(3).standard_normal((30, 80)).T)
    s2 = triger.sort_result_XandY(X, Y, blocks)
    assert set(s2[0][1].tolist()) == set(blocks[0][1].tolist())


def test_greedy_3step():
    rng = np.random.default_rng(4)
    N, m, n = 150, 80, 120
    F = rng.standard_normal(N)
    X = rng.standard_normal((N, m)); Y = rng.standard_normal((N, n))
    X[:, :25] += 1.5 * F[:, None]; Y[:, :40] += 1.5 * F[:, None]
    R = np.corrcoef(np.hstack([X, Y]).T)
    P = np.abs(R); np.fill_diagonal(P, 0)
    blocks = triger.greedy_3step(P, R, m, n, 1.1, 1.8, 1.8, X_size=(10, 200), Y_size=(10, 200))
    assert blocks
    assert len(set(blocks[0][0].tolist()) & set(range(25))) >= 20


def test_plots_run():
    W = planted()
    blocks = triger.sort_result(W, triger.greedy_peeling_XY_all(W, 1.5))
    X = np.corrcoef(np.random.default_rng(5).standard_normal((40, 60)).T)
    Y = np.corrcoef(np.random.default_rng(6).standard_normal((40, 80)).T)
    triger.plot_all_result(W, blocks, full=True)
    triger.plot3in1_v1(X, Y, W, blocks, line=True)
