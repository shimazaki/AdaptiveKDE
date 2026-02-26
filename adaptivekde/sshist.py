import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def sshist(x, N=range(2, 501), SN=30):
    """
    Returns the optimal number of bins in a histogram used for density
    estimation.

    Optimization principle is to minimize expected L2 loss function between
    the histogram and an unknown underlying density function.
    An assumption made is merely that samples are drawn from the density
    independently each other.

    The optimal binwidth D* is obtained as a minimizer of the formula,
    (2K-V) / D^2,
    where K and V are mean and variance of sample counts across bins with width
    D. Optimal number of bins is given as (max(x) - min(x)) / D.

    Parameters
    ----------
    x : array_like
        One-dimensional data to fit histogram to.
    N : array_like, optional
        Array containing number of histogram bins to evaluate for fit.
        Default value = 500.
    SN : double, optional
        Scalar natural number defining number of bins for shift-averaging.

    Returns
    -------
    optN : int
        Optimal number of bins to represent the data in X
    optD : double
        Optimal width of bins
    edges : array_like
        Edges of optimized bins
    N : double
        Maximum number of bins to be evaluated. Default value = 500.
    C : array_like
        Cost function C[i] of evaluating histogram fit with N[i] bins

    See Also
    --------
    sskernel, ssvkernel

    References
    ----------
    .. [1] H. Shimazaki and S. Shinomoto, "A method for selecting the bin size
           of a time histogram," in  Neural Computation 19(6), 1503-1527, 2007
           http://dx.doi.org/10.1162/neco.2007.19.6.1503
    """

    # determine range of input 'x'
    x_min = np.min(x)
    x_max = np.max(x)

    # get smallest difference 'dx' between all pairwise samples
    buf = np.abs(np.diff(np.sort(x)))
    dx = np.min(buf[buf > 0])

    # setup bins to evaluate
    N_MIN = 2
    N_MAX = int(min(np.floor((x_max - x_min) / (2*dx)), max(N)))
    N = range(N_MIN, N_MAX+1)
    D = (x_max - x_min) / N

    # compute cost function over each possible number of bins
    x_sorted = np.sort(x)
    if _HAS_NUMBA:
        C = _sshist_cost_numba(x_sorted, x_min, x_max, N_MIN, N_MAX, SN)
    else:
        C = _sshist_cost_numpy(x_sorted, x_min, x_max, N, D, SN)

    # get bin count that minimizes cost C
    idx = np.argmin(C)
    optN = N[idx]
    optD = D[idx]
    edges = np.linspace(x_min, x_max, optN)

    return optN, optD, edges, C, N


def _sshist_cost_numpy(x_sorted, x_min, x_max, N, D, SN):
    """Vectorized NumPy cost computation (no numba)."""
    Cs = np.zeros((len(N), SN))
    for i, n in enumerate(N):
        shift = np.linspace(0, D[i], SN)
        lo = x_min + shift - D[i] / 2
        hi = x_max + shift - D[i] / 2
        n_edges = N[i] + 1
        frac = np.linspace(0, 1, n_edges)
        all_edges = lo[:, np.newaxis] + frac[np.newaxis, :] * (hi - lo)[:, np.newaxis]
        ss = np.searchsorted(x_sorted, all_edges.ravel())
        counts = np.diff(ss.reshape(SN, n_edges), axis=1)
        k = counts.mean(axis=1)
        v = np.sum((counts - k[:, np.newaxis])**2, axis=1) / N[i]
        Cs[i, :] = (2 * k - v) / D[i]**2
    return Cs.mean(axis=1)


def _make_numba_kernel():
    """Create numba-JIT compiled cost function (called once at import)."""
    @njit(cache=True)
    def _cost(x_sorted, x_min, x_max, N_MIN, N_MAX, SN):
        T = x_max - x_min
        n_range = N_MAX - N_MIN + 1
        C = np.zeros(n_range)
        for i in range(n_range):
            n = N_MIN + i
            D = T / n
            cost_sum = 0.0
            for p in range(SN):
                sh = p * D / (SN - 1) if SN > 1 else 0.0
                # match np.linspace edge computation exactly
                base = x_min + sh - D / 2
                end = x_max + sh - D / 2
                span = end - base
                # count events in each bin using searchsorted
                k_sum = 0.0
                v_sum = 0.0
                prev_idx = np.searchsorted(x_sorted, base)
                for b in range(1, n + 1):
                    edge = base + span * b / n
                    cur_idx = np.searchsorted(x_sorted, edge)
                    count = cur_idx - prev_idx
                    k_sum += count
                    v_sum += count * count
                    prev_idx = cur_idx
                mean_k = k_sum / n
                var = v_sum / n - mean_k * mean_k
                cost_sum += (2 * mean_k - var) / (D * D)
            C[i] = cost_sum / SN
        return C
    return _cost


if _HAS_NUMBA:
    _sshist_cost_numba = _make_numba_kernel()
