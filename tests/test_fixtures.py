"""Compares the package with the reference outputs described in
tests/fixtures/README.md."""
import csv
from pathlib import Path

import numpy as np

import triger

FIXTURES = Path(__file__).parent / "fixtures"


def read_matrix(name):
    return np.loadtxt(FIXTURES / name, delimiter=",")


def read_blocks(name, sides):
    blocks = {}
    with open(FIXTURES / name, newline="") as fh:
        for row in csv.DictReader(fh):
            blk = blocks.setdefault(int(row["block"]), {s: [] for s in sides})
            blk[row["side"]].append(int(row["index"]) - 1)
    return [blocks[k] for k in sorted(blocks)]


def read_density(name):
    with open(FIXTURES / name, newline="") as fh:
        return [float(row["density"]) for row in csv.DictReader(fh)]


def test_xy_all_matches_reference():
    blocks = triger.greedy_peeling_XY_all(read_matrix("xy.csv"), 1.5,
                                          max_blocks=4)
    expected = read_blocks("xy_blocks.csv", "XY")
    assert len(blocks) == len(expected)
    for b, e in zip(blocks, expected):
        assert b.X.tolist() == e["X"] and b.Y.tolist() == e["Y"]
    np.testing.assert_allclose([b.density for b in blocks],
                               read_density("xy_density.csv"), rtol=1e-6)


def test_x_all_matches_reference():
    comms = triger.greedy_peeling_X_all(read_matrix("x.csv"), 1.5)
    expected = read_blocks("x_blocks.csv", "X")
    assert [c.X.tolist() for c in comms] == [e["X"] for e in expected]
    np.testing.assert_allclose([c.density for c in comms],
                               read_density("x_density.csv"), rtol=1e-6)


def test_3step_matches_reference():
    modules = triger.greedy_3step(read_matrix("three_step_P.csv"),
                                  read_matrix("three_step_R.csv"), 30, 40,
                                  lambda1=1.1, lambda2=1.5, lambda3=1.5,
                                  X_size=(5, 10), Y_size=(5, 15),
                                  max_blocks=4)
    expected = read_blocks("three_step_blocks.csv", "XY")
    assert len(modules) == len(expected)
    for b, e in zip(modules, expected):
        assert b.X.tolist() == e["X"] and b.Y.tolist() == e["Y"]
