import numpy as np


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
    Cs = np.zeros((len(N), SN))
    for i, n in enumerate(N):  # loop over number of bins
        shift = np.linspace(0, D[i], SN)
        # batch all SN shifts: build (SN, n+1) edges, flatten for searchsorted
        lo = x_min + shift - D[i] / 2       # (SN,) left edges
        hi = x_max + shift - D[i] / 2       # (SN,) right edges
        n_edges = N[i] + 1
        # linspace for each shift row via broadcasting
        frac = np.linspace(0, 1, n_edges)    # (n_edges,)
        all_edges = lo[:, np.newaxis] + frac[np.newaxis, :] * (hi - lo)[:, np.newaxis]
        # flatten, searchsorted, reshape, diff → counts (SN, n)
        ss = np.searchsorted(x_sorted, all_edges.ravel())
        counts = np.diff(ss.reshape(SN, n_edges), axis=1)  # (SN, n)
        k = counts.mean(axis=1)              # (SN,)
        v = np.sum((counts - k[:, np.newaxis])**2, axis=1) / N[i]
        Cs[i, :] = (2 * k - v) / D[i]**2

    # average over shift window
    C = Cs.mean(axis=1)

    # get bin count that minimizes cost C
    idx = np.argmin(C)
    optN = N[idx]
    optD = D[idx]
    edges = np.linspace(x_min, x_max, optN)

    return optN, optD, edges, C, N
