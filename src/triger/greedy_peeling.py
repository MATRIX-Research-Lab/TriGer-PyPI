"""
Greedy peeling for TriGer: cross-omics (X-Y) and within-omics (symmetric)
peeling, block sorting, and greedy_3step, the full pipeline on precomputed
association matrices. All indices are 0-based.
"""
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class Block(NamedTuple):
    """A cross-omics block: X (row) indices, Y (column) indices and the
    penalized density of the block."""
    X: np.ndarray
    Y: np.ndarray
    density: float


class Community(NamedTuple):
    """A within-omics community: feature indices and penalized density."""
    X: np.ndarray
    density: float


# ----------------------------------------------------------------------
# Input checks
# ----------------------------------------------------------------------
def _as_matrix(x, name):
    try:
        x = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"`{name}` must be a numeric matrix") from None
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"`{name}` must be a matrix with at least one row "
                         "and one column")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"`{name}` must not contain missing or infinite "
                         "values")
    return x


def _check_symmetric(x, name):
    if x.shape[0] != x.shape[1] or not np.allclose(x, x.T, rtol=0,
                                                    atol=1e-8):
        raise ValueError(f"`{name}` must be symmetric")


def _check_number(x, name):
    if (isinstance(x, bool) or not isinstance(x, (int, float, np.number))
            or not np.isfinite(x)):
        raise ValueError(f"`{name}` must be a finite number")


def _check_count(x, name, minimum=1):
    if (isinstance(x, bool) or not isinstance(x, (int, np.integer))
            or x < minimum):
        raise ValueError(f"`{name}` must be an integer of at least {minimum}")


def _check_size(x, name):
    try:
        lo, hi = x
    except (TypeError, ValueError):
        raise ValueError(f"`{name}` must be (min, max)") from None
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        raise ValueError(f"`{name}` must be (min, max) with min <= max")


# ----------------------------------------------------------------------
# Step 1: cross-omics (X x Y) peeling
# ----------------------------------------------------------------------
def greedy_peeling_XY_one(Wp, lam):
    """Peel one dense block from a non-negative cross-omics matrix.

    Greedily removes the X feature (row) or Y feature (column) with the
    weakest remaining association and keeps the intermediate submatrix with
    the highest penalized density, ``sum / (|X| * |Y|) ** (lam / 2)``. A row
    is removed when its sum, scaled by the current ratio ``|X| / |Y|``, is
    at most the smallest column sum; otherwise a column is removed.

    Parameters
    ----------
    Wp : array_like, shape (m, n)
        Non-negative cross-omics associations (X features in rows, Y
        features in columns), for example -log10 p-values.
    lam : float
        Size penalty, typically in [1, 2]. Higher values give smaller,
        denser blocks.

    Returns
    -------
    density : float
        Penalized density of the block.
    X, Y : ndarray
        Row and column indices of the block.
    """
    Wp = _as_matrix(Wp, "Wp")
    _check_number(lam, "lam")
    m, n = Wp.shape
    C = Wp.sum(axis=0)
    R = Wp.sum(axis=1)
    rem_X = np.zeros(m, dtype=bool)
    rem_Y = np.zeros(n, dtype=bool)
    best_rem_X, best_rem_Y = rem_X.copy(), rem_Y.copy()
    len_X, len_Y = m, n
    W_density = 0.0

    for _ in range(m + n):
        Cc = np.where(rem_Y, np.inf, C)
        Rr = np.where(rem_X, np.inf, R)
        IndT = int(np.argmin(Cc))
        dT = Cc[IndT]
        IndS = int(np.argmin(Rr))
        dS = Rr[IndS]
        if not (np.isfinite(dT) and np.isfinite(dS)):
            break
        if (len_X / len_Y) * dS <= dT:        # remove a row (X feature)
            rem_X[IndS] = True
            C = C - Wp[IndS, :]
            C[rem_Y] = 0.0
            Wp_sum = C.sum()
            len_X -= 1
        else:                                  # remove a column (Y feature)
            rem_Y[IndT] = True
            R = R - Wp[:, IndT]
            R[rem_X] = 0.0
            Wp_sum = R.sum()
            len_Y -= 1
        if len_X <= 0 or len_Y <= 0:
            break
        score = Wp_sum / (len_X * len_Y) ** (lam / 2.0)
        if np.isfinite(score) and score > W_density:
            W_density = float(score)
            best_rem_X, best_rem_Y = rem_X.copy(), rem_Y.copy()

    return W_density, np.flatnonzero(~best_rem_X), np.flatnonzero(~best_rem_Y)


def greedy_peeling_XY_all(XY, lam=1.5, max_blocks=5) -> List[Block]:
    """Peel dense blocks from a cross-omics matrix repeatedly.

    Applies :func:`greedy_peeling_XY_one` repeatedly, removing the rows and
    columns of each accepted block before the next round. A block is
    accepted while its penalized density exceeds that of the full matrix, up
    to ``max_blocks`` blocks.

    Parameters
    ----------
    XY : array_like, shape (m, n)
        Non-negative cross-omics association matrix.
    lam : float
        Size penalty.
    max_blocks : int
        Maximum number of blocks returned.

    Returns
    -------
    list of Block
        Blocks with indices into the rows (``X``) and columns (``Y``) of XY.
    """
    XY = _as_matrix(XY, "XY")
    _check_number(lam, "lam")
    _check_count(max_blocks, "max_blocks")
    m, n = XY.shape
    org_density = XY.sum() / (m * n) ** (lam / 2.0)
    remain_X, remain_Y = np.arange(m), np.arange(n)
    result: List[Block] = []
    while len(result) < max_blocks and len(remain_X) and len(remain_Y):
        dens, sX, sY = greedy_peeling_XY_one(XY[np.ix_(remain_X, remain_Y)],
                                             lam)
        if dens <= org_density or len(sX) == 0 or len(sY) == 0:
            break
        blk = Block(remain_X[sX], remain_Y[sY], dens)
        result.append(blk)
        remain_X = np.setdiff1d(remain_X, blk.X)
        remain_Y = np.setdiff1d(remain_Y, blk.Y)
    return result


# ----------------------------------------------------------------------
# Steps 2 and 3: within-omics (symmetric) peeling
# ----------------------------------------------------------------------
def greedy_peeling_X_one(Wp, lam):
    """Peel one dense community from a symmetric within-omics matrix.

    Greedily removes the feature with the smallest remaining sum and keeps
    the intermediate subset with the highest penalized density,
    ``sum / size ** lam``.

    Parameters
    ----------
    Wp : array_like, shape (m, m)
        Symmetric within-omics associations, for example signed
        correlations.
    lam : float
        Size penalty.

    Returns
    -------
    density : float
        Penalized density of the community.
    X : ndarray
        Indices of the community.
    """
    Wp = _as_matrix(Wp, "Wp")
    _check_symmetric(Wp, "Wp")
    _check_number(lam, "lam")
    m = Wp.shape[0]
    C = Wp.sum(axis=0)
    rem = np.zeros(m, dtype=bool)
    best_rem = rem.copy()
    W_density = 0.0
    for i in range(1, m + 1):
        Ind = int(np.argmin(np.where(rem, np.inf, C)))
        rem[Ind] = True
        C = C - Wp[Ind, :]
        C[rem] = 0.0
        if i == m:
            break
        score = C.sum() / (m - i) ** lam
        if np.isfinite(score) and score > W_density:
            W_density = float(score)
            best_rem = rem.copy()
    return W_density, np.flatnonzero(~best_rem)


def greedy_peeling_X_all(Wp, lam, min_size=1) -> List[Community]:
    """Peel dense communities from a within-omics matrix repeatedly.

    Applies :func:`greedy_peeling_X_one` repeatedly, removing each accepted
    community before the next round, while the community's penalized
    density exceeds that of the full matrix.

    Parameters
    ----------
    Wp : array_like, shape (m, m)
        Symmetric within-omics association matrix.
    lam : float
        Size penalty.
    min_size : int
        Smallest community returned. Communities with fewer features are
        peeled but not returned. The default, 1, returns every community.

    Returns
    -------
    list of Community
        Communities with indices into the rows of Wp.
    """
    Wp = _as_matrix(Wp, "Wp")
    _check_symmetric(Wp, "Wp")
    _check_number(lam, "lam")
    _check_count(min_size, "min_size")
    m = Wp.shape[0]
    org_density = Wp.sum() / m ** lam
    remain = np.arange(m)
    result: List[Community] = []
    while len(remain):
        dens, sel = greedy_peeling_X_one(Wp[np.ix_(remain, remain)], lam)
        if dens <= org_density or len(sel) == 0:
            break
        comm = Community(remain[sel], dens)
        result.append(comm)
        remain = np.setdiff1d(remain, comm.X)
    return [c for c in result if len(c.X) >= min_size]


# ----------------------------------------------------------------------
# Ordering features within blocks
# ----------------------------------------------------------------------
def sort_result(Wp, blocks, symmetric: Optional[bool] = None):
    """Order the features of each block by their within-block sums in Wp.

    For a cross-omics Wp, rows and columns are ordered by the absolute
    values of their sums. For a within-omics (symmetric) Wp, pass
    communities and ``symmetric=True`` (the default for square matrices).

    Returns
    -------
    list
        ``blocks`` with the features of each block reordered; records keep
        their type (Block, Community or tuple).
    """
    Wp = np.asarray(Wp, dtype=np.float64)
    if symmetric is None:
        symmetric = Wp.shape[0] == Wp.shape[1]
    out = []
    for blk in blocks:
        if symmetric:
            idx = np.asarray(blk[0], dtype=int)
            sums = Wp[np.ix_(idx, idx)].sum(axis=1)
            new = (idx[np.argsort(-sums, kind="stable")],) + tuple(blk[1:])
        else:
            xi = np.asarray(blk[0], dtype=int)
            yi = np.asarray(blk[1], dtype=int)
            sub = Wp[np.ix_(xi, yi)]
            ox = np.argsort(-np.abs(sub.sum(axis=1)), kind="stable")
            oy = np.argsort(-np.abs(sub.sum(axis=0)), kind="stable")
            new = (xi[ox], yi[oy]) + tuple(blk[2:])
        out.append(type(blk)(*new) if hasattr(blk, "_fields") else new)
    return out


def sort_result_XandY(X, Y, blocks):
    """Order each block's X features by their sums within ``X[xi, xi]`` and
    its Y features by their sums within ``Y[yi, yi]``."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    out = []
    for blk in blocks:
        xi = np.asarray(blk[0], dtype=int)
        yi = np.asarray(blk[1], dtype=int)
        ox = np.argsort(-X[np.ix_(xi, xi)].sum(axis=0), kind="stable")
        oy = np.argsort(-Y[np.ix_(yi, yi)].sum(axis=0), kind="stable")
        new = (xi[ox], yi[oy]) + tuple(blk[2:])
        out.append(type(blk)(*new) if hasattr(blk, "_fields") else new)
    return out


# ----------------------------------------------------------------------
# The full pipeline on precomputed matrices
# ----------------------------------------------------------------------
def greedy_3step(P, R, m, n, lambda1=1.5, lambda2=1.5, lambda3=1.5,
                 X_size: Tuple[float, float] = (10, 200),
                 Y_size: Tuple[float, float] = (10, 200),
                 min_community_size=2, keep_unrefined_if_empty=True,
                 max_blocks=5) -> List[Block]:
    """Detect cross-omics modules from association matrices.

    Step 1 peels dense blocks from the cross-omics part of P with
    :func:`greedy_peeling_XY_all`. Blocks with ``X_size[0]`` or fewer X
    features, or ``Y_size[0]`` or fewer Y features, are dropped. In steps 2
    and 3, the X and Y sides of blocks with more than ``X_size[1]`` or
    ``Y_size[1]`` features are refined by within-omics peeling of the signed
    correlations in R: the side is replaced by the communities with at least
    ``min_community_size`` features, each ordered by within-community sums.
    :func:`triger.triger` computes P and R from data and calls this
    function.

    Parameters
    ----------
    P : array_like, shape (m + n, m + n)
        Non-negative association matrix over the X and Y features (for
        example -log10 p-values of their correlations); only ``P[:m, m:]``
        is used.
    R : array_like, shape (m + n, m + n)
        Signed correlation matrix over the X and Y features, used for
        refinement.
    m, n : int
        Numbers of X and Y features.
    lambda1 : float
        Cross-omics size penalty (step 1).
    lambda2, lambda3 : float
        Within-omics size penalties for X and Y (steps 2 and 3).
    X_size, Y_size : (min, max)
        Blocks with ``min`` or fewer features on either side are dropped,
        and sides with more than ``max`` features are refined.
    min_community_size : int
        Smallest within-omics community kept during refinement.
    keep_unrefined_if_empty : bool
        If refinement keeps no community, keep that side of the block
        unrefined (True) or empty it (False).
    max_blocks : int
        Maximum number of blocks found in step 1.

    Returns
    -------
    list of Block
        Modules with ``X`` in ``0..m-1``, ``Y`` in ``0..n-1``, and
        ``density`` the penalized density of the step-1 block before
        refinement.
    """
    P = _as_matrix(P, "P")
    R = _as_matrix(R, "R")
    _check_count(m, "m")
    _check_count(n, "n")
    if P.shape != (m + n, m + n):
        raise ValueError("`P` must be (m + n) x (m + n)")
    if R.shape != (m + n, m + n):
        raise ValueError("`R` must be (m + n) x (m + n)")
    _check_size(X_size, "X_size")
    _check_size(Y_size, "Y_size")
    _check_count(min_community_size, "min_community_size")

    blocks = greedy_peeling_XY_all(P[:m, m:], lambda1, max_blocks)
    blocks = [b for b in blocks
              if len(b.X) > X_size[0] and len(b.Y) > Y_size[0]]

    def refine(idx, offset, lam):
        sub = R[np.ix_(offset + idx, offset + idx)]
        sub = (sub + sub.T) / 2.0
        comm = greedy_peeling_X_all(sub, lam, min_size=min_community_size)
        if not comm:
            return idx if keep_unrefined_if_empty else idx[:0]
        comm = sort_result(sub, comm, symmetric=True)
        return idx[np.concatenate([c.X for c in comm])]

    out = []
    for b in blocks:
        xi, yi = b.X, b.Y
        if len(xi) > X_size[1]:
            xi = refine(xi, 0, lambda2)
        if len(yi) > Y_size[1]:
            yi = refine(yi, m, lambda3)
        out.append(Block(xi, yi, b.density))
    return out
