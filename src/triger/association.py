"""
TriGer on data: association matrices for two omics layers and the full
pipeline. All indices are 0-based.
"""
from typing import List, NamedTuple, Optional

import numpy as np
from scipy import stats

from .greedy_peeling import _as_matrix, greedy_3step


class Association(NamedTuple):
    """Association matrices over the features of two omics layers: signed
    correlations ``R``, -log10 p-values ``P``, feature names, and the
    numbers of X and Y features."""
    R: np.ndarray
    P: np.ndarray
    names: List[str]
    m: int
    n: int


class Module(NamedTuple):
    """A cross-omics module: X and Y indices, the penalized density of the
    step-1 block, and the X and Y feature names."""
    X: np.ndarray
    Y: np.ndarray
    density: float
    X_names: List[str]
    Y_names: List[str]


class TrigerResult(NamedTuple):
    """Output of :func:`triger`: the modules and the association matrices."""
    modules: List[Module]
    R: np.ndarray
    P: np.ndarray
    names: List[str]
    m: int
    n: int


def _feature_names(x, prefix):
    cols = getattr(x, "columns", None)
    if cols is not None:
        return [str(c) for c in cols]
    return [f"{prefix}{i}" for i in range(np.shape(x)[1])]


def association_matrices(X, Y, method="pearson") -> Association:
    """Association matrices for two omics layers.

    Computes the signed correlation matrix and the matrix of -log10
    two-sided p-values over the features of two omics layers measured on
    the same samples. These are the inputs ``R`` and ``P`` of
    :func:`triger.greedy_3step`.

    The p-value of a correlation coefficient r estimated from N samples is
    computed from ``t = |r| * sqrt((N - 2) / (1 - r**2))`` with N - 2
    degrees of freedom, as in ``scipy.stats.pearsonr`` (for Spearman
    correlations, the same t approximation). ``r**2`` is capped at
    ``1 - 1e-12`` so that perfectly correlated features receive a large
    finite value, and the diagonal of P is set to 0.

    Parameters
    ----------
    X : array_like or pandas.DataFrame, shape (N, m)
        First omics layer, samples in rows and features in columns.
    Y : array_like or pandas.DataFrame, shape (N, n)
        Second omics layer, with the same samples in the same order.
    method : {"pearson", "spearman"}
        Correlation coefficient.

    Returns
    -------
    Association
        ``R`` and ``P``, both ``(m + n, m + n)`` with the m features of X
        first; ``names``, the column names of X and Y for DataFrames and
        ``X0, X1, ...`` and ``Y0, Y1, ...`` otherwise; ``m`` and ``n``.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("`method` must be 'pearson' or 'spearman'")
    names = _feature_names(X, "X") + _feature_names(Y, "Y")
    X = _as_matrix(X, "X")
    Y = _as_matrix(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("`X` and `Y` must have the same number of samples "
                         "(rows)")
    N = X.shape[0]
    if N < 3:
        raise ValueError("at least 3 samples are required")
    XY = np.hstack([X, Y])
    constant = np.all(XY == XY[0], axis=0)
    if constant.any():
        bad = [names[i] for i in np.flatnonzero(constant)[:5]]
        raise ValueError("features with zero variance: " + ", ".join(bad))
    if method == "spearman":
        XY = np.apply_along_axis(stats.rankdata, 0, XY)
    R = np.corrcoef(XY, rowvar=False)
    t = np.abs(R) * np.sqrt((N - 2) / (1 - np.minimum(R ** 2, 1 - 1e-12)))
    log_p = np.log(2) + stats.t.logsf(t, df=N - 2)
    P = -log_p / np.log(10)
    np.fill_diagonal(P, 0.0)
    return Association(R, P, names, X.shape[1], Y.shape[1])


def _match_samples(X, Y):
    ix = getattr(X, "index", None)
    iy = getattr(Y, "index", None)
    if ix is None or iy is None:
        return Y
    if (not ix.is_unique or not iy.is_unique
            or set(ix) != set(iy)):
        raise ValueError("the sample names (index) of `X` and `Y` must be "
                         "unique and identical")
    return Y.loc[ix]


def triger(X, Y, lambda1=1.5, lambda2=1.5, lambda3=1.5, X_size=(10, 200),
           Y_size=(10, 200), method="pearson", **kwargs) -> TrigerResult:
    """Detect cross-omics modules with TriGer.

    Runs the TriGer pipeline on two omics layers measured on the same
    samples. :func:`association_matrices` computes the cross- and
    within-omics correlations and their -log10 p-values, and
    :func:`triger.greedy_3step` finds blocks of X and Y features that are
    densely associated across the two layers and refines large blocks by
    within-omics peeling.

    When X and Y are both pandas DataFrames, samples are matched by index
    and Y is reordered to the sample order of X. Otherwise, the samples must
    be in the same order.

    Parameters
    ----------
    X, Y : array_like or pandas.DataFrame
        The two omics layers, samples in rows and features in columns.
    lambda1 : float
        Cross-omics size penalty (step 1). Higher values give smaller,
        denser blocks.
    lambda2, lambda3 : float
        Within-omics size penalties for X and Y (steps 2 and 3).
    X_size, Y_size : (min, max)
        Passed to :func:`triger.greedy_3step`: blocks with ``min`` or fewer
        features on either side are dropped, and sides with more than
        ``max`` features are refined.
    method : {"pearson", "spearman"}
        Correlation coefficient.
    **kwargs
        Further arguments to :func:`triger.greedy_3step`:
        ``min_community_size``, ``keep_unrefined_if_empty``, ``max_blocks``.

    Returns
    -------
    TrigerResult
        ``modules``, a list of :class:`Module` records with indices into
        the features of X and Y and their names, and the association
        matrices ``R``, ``P``, ``names``, ``m`` and ``n``.
    """
    Y = _match_samples(X, Y)
    assoc = association_matrices(X, Y, method=method)
    m, n = assoc.m, assoc.n
    blocks = greedy_3step(assoc.P, assoc.R, m, n, lambda1=lambda1,
                          lambda2=lambda2, lambda3=lambda3, X_size=X_size,
                          Y_size=Y_size, **kwargs)
    x_names, y_names = assoc.names[:m], assoc.names[m:]
    modules = [Module(b.X, b.Y, b.density, [x_names[i] for i in b.X],
                      [y_names[j] for j in b.Y]) for b in blocks]
    return TrigerResult(modules, *assoc)
