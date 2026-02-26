import numpy as np


def ssvkernel(x, tin=None, M=80, nbs=100, WinFunc='Boxcar'):
    """
    Generates a locally adaptive kernel-density estimate for one-dimensional
    data.

    The user provides a one-dimensional vector of samples drawn from some
    underlying unknown distribution, and optionally the values where they want
    to estimate the probability density of that distribution. The algorithm
    solves an optimization problem to identify variable bandwidths across the
    domain where the data is provided.

    The optimization is based on a principle of minimizing expected L2 loss
    function between the kernel estimate and an unknown underlying density
    function. An assumption is merely that samples are drawn from the density
    independently of each other.

    The locally adaptive bandwidth is obtained by iteratively computing optimal
    fixed-size bandwidths wihtihn local intervals. The optimal bandwidths are
    selected such that they are selected in the intervals that are gamma times
    larger than the optimal bandwidths themselves. The paramter gamma is
    optimized by minimizing the L2 risk estimate.

    Parameters
    ----------
    x : array_like
        The one-dimensional samples drawn from the underlying density
    tin : array_like, optional
        The values where the density estimate is to be evaluated in generating
        the output 'y'. Default value = None.
    M : int, optional
        The number of window sizes to evaluate. Default value = 80.
    nbs : int, optional
        The number of bootstrap samples to use in estimating the [0.05, 0.95]
        confidence interval of the output 'y'.
    WinFunc : string, optional
        The type of window function to use in estimating local bandwidth.
        Choose from one of 'Boxcar', 'Laplace', 'Cauchy' and 'Gauss'. Default
        value = 'Gauss'.

    Returns
    -------
    y : array_like
        The estimated density, evaluated at points t / tin.
    t : array_like
        The points where the density estimate 'y' is evaluated.
    optw : array_like
        The optimal local kernel bandwidths at 't'.
    gs : array_like
        The stiffness constants of the variables bandwidths evaluated.
    C : array_like
        Cost functions associated with stiffness constraints.
    confb95 : array_like
        The 5% and 95% confidence interval of the kernel density estimate 'y'.
        Has dimensions 2 x len(y). confb95[0,:] corresponds to the 5% interval,
        and confb95[1,:] corresponds to the 95% interval.
    yb : array_like
        The bootstrap samples used in estimating confb95. Each row corresponds
        to one bootstrap sample.

    See Also
    --------
    sshist, sskernel

    References
    ----------
    .. [1] H. Shimazaki and S. Shinomoto, "Kernel Bandwidth Optimization in 
           Spike Rate Estimation," in Journal of Computational Neuroscience 
           29(1-2): 171–182, 2010 http://dx.doi.org/10.1007/s10827-009-0180-4
    """

    # set argument 't' if not provided
    if tin is None:
        T = np.max(x) - np.min(x)
        dx = np.sort(np.diff(np.sort(x)))
        dt_samp = dx[np.nonzero(dx)][0]
        tin = np.linspace(np.min(x), np.max(x), int(min(np.ceil(T / dt_samp), 1e3)))
        t = tin
        x_ab = x[(x >= min(tin)) & (x <= max(tin))]
    else:
        T = np.max(x) - np.min(x)
        x_ab = x[(x >= min(tin)) & (x <= max(tin))]
        dx = np.sort(np.diff(np.sort(x)))
        dt_samp = dx[np.nonzero(dx)][0]
        if dt_samp > min(np.diff(tin)):
            t = np.linspace(min(tin), max(tin), int(min(np.ceil(T / dt_samp), 1e3)))
        else:
            t = tin

    # calculate delta t
    dt = np.min(np.diff(t))

    # create the finest histogram
    thist = np.concatenate((t, (t[-1]+dt)[np.newaxis]))
    y_hist = np.histogram(x_ab, thist-dt/2)[0] / dt
    L = y_hist.size
    N = np.sum(y_hist * dt).astype(float)

    # initialize window sizes
    W = logexp(np.linspace(ilogexp(5 * dt), ilogexp(T), M))

    # compute local cost functions
    c = np.zeros((M, L))
    for j in range(M):
        w = W[j]
        yh = fftkernel(y_hist, w / dt)
        c[j, :] = yh**2 - 2 * yh * y_hist + 2 / (2 * np.pi)**0.5 / w * y_hist

    # initialize optimal ws — batch FFTs by grouping on padded size n
    optws = np.zeros((M, L))

    # Group outer iterations by FFT size
    fft_groups = {}
    for i in range(M):
        w = W[i] / dt
        n_fft = int(2 ** np.ceil(np.log2(L + 3 * w)))
        fft_groups.setdefault(n_fft, []).append((i, w))

    for n_fft, group in fft_groups.items():
        # Forward FFT of all c rows at this padded size — done ONCE per group
        C_fft = np.fft.rfft(c, n_fft, axis=1)  # shape (M, n_fft//2+1)
        f = _get_rfreq(n_fft)
        t_freq = 2 * np.pi * f

        for i, w in group:
            # Compute window kernel for this bandwidth
            if WinFunc == 'Boxcar':
                a_bx = 12**0.5 * w
                K = np.zeros(len(t_freq))
                K[0] = 1
                K[1:] = 2 * np.sin(a_bx * t_freq[1:] / 2) / (a_bx * t_freq[1:])
            elif WinFunc == 'Laplace':
                K = 1 / (1 + (w * 2 * np.pi * f)**2 / 2)
            elif WinFunc == 'Cauchy':
                K = np.exp(-w * np.abs(2 * np.pi * f))
            else:
                K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)

            # Batched IFFT: apply kernel to all M rows at once
            C_local = np.fft.irfft(C_fft * K[np.newaxis, :], n_fft, axis=1)[:, :L]
            n_idx = np.argmin(C_local, axis=0)
            optws[i, :] = W[n_idx]

    # golden section search for stiffness parameter of variable bandwidths
    k = 0
    gs = np.zeros((30, 1))
    C = np.zeros((30, 1))
    tol = 1e-5
    a = 1e-12
    b = 1
    phi = (5**0.5 + 1) / 2
    c1 = (phi - 1) * a + (2 - phi) * b
    c2 = (2 - phi) * a + (phi - 1) * b
    f1 = CostFunction(y_hist, N, t, dt, optws, W, WinFunc, c1)[0]
    f2 = CostFunction(y_hist, N, t, dt, optws, W, WinFunc, c2)[0]
    while (np.abs(b-a) > tol * (abs(c1) + abs(c2))) & (k < 30):
        if f1 < f2:
            b = c2
            c2 = c1
            c1 = (phi - 1) * a + (2 - phi) * b
            f2 = f1
            f1, yv1, optwp1 = CostFunction(y_hist, N, t, dt, optws, W,
                                           WinFunc, c1)
            yopt = yv1 / np.sum(yv1 * dt)
            optw = optwp1
        else:
            a = c1
            c1 = c2
            c2 = (2 - phi) * a + (phi - 1) * b
            f1 = f2
            f2, yv2, optwp2 = CostFunction(y_hist, N, t, dt, optws, W,
                                           WinFunc, c2)
            yopt = yv2 / np.sum(yv2 * dt)
            optw = optwp2

        # capture estimates and increment iteration counter
        gs[k] = c1
        C[k] = f1
        k = k + 1

    # discard unused entries in gs, C
    gs = gs[0:k]
    C = C[0:k]

    # estimate confidence intervals by bootstrapping
    nbs = np.asarray(nbs)
    yb = np.zeros((nbs, tin.size))
    thist = np.concatenate((t, (t[-1]+dt)[np.newaxis]))
    bins = thist - dt / 2
    inv_2pi2 = 1 / (2 * np.pi)**2
    optw_sq2 = 2 * optw**2                       # pre-compute for Gauss
    for i in range(nbs):
        Nb = np.random.poisson(lam=N)
        idx = np.random.randint(0, N, Nb)
        xb = x_ab[idx]
        y_histb = np.histogram(xb, bins)[0]
        idx_nz = y_histb.nonzero()
        y_histb_nz = y_histb[idx_nz]
        t_nz = t[idx_nz]
        # vectorized L×nnz balloon estimator
        t_diff_nz = t[:, np.newaxis] - t_nz[np.newaxis, :]
        G = (inv_2pi2 / optw[:, np.newaxis]
             * np.exp(-t_diff_nz**2 / optw_sq2[:, np.newaxis]))
        yb_buf = np.sum(y_histb_nz[np.newaxis, :] * G, axis=1) / Nb
        yb_buf = yb_buf / np.sum(yb_buf * dt)
        yb[i, :] = np.interp(tin, t, yb_buf)
    ybsort = np.sort(yb, axis=0)
    y95b = ybsort[int(np.floor(0.05 * nbs)), :]
    y95u = ybsort[int(np.floor(0.95 * nbs)), :]
    confb95 = np.concatenate((y95b[np.newaxis], y95u[np.newaxis]), axis=0)

    # return outputs
    y = np.interp(tin, t, yopt)
    optw = np.interp(tin, t, optw)
    t = tin

    return y, t, optw, gs, C, confb95, yb


def CostFunction(y_hist, N, t, dt, optws, WIN, WinFunc, g):

    L = y_hist.size
    M = WIN.size

    # --- vectorized optwv: replaces per-k loop ---
    gs_all = optws / WIN[:, np.newaxis]          # (M, L)
    gs_max = np.max(gs_all, axis=0)              # (L,)
    gs_min = np.min(gs_all, axis=0)              # (L,)

    optwv = np.full(L, np.max(WIN))              # default: g < min(gs)
    mask_high = g > gs_max
    optwv[mask_high] = np.min(WIN)
    mask_mid = ~mask_high & (g >= gs_min)
    if np.any(mask_mid):
        ge_mask = gs_all[:, mask_mid] >= g       # (M, count)
        row_idx = np.arange(M)[:, np.newaxis]
        max_idx = np.max(np.where(ge_mask, row_idx, -1), axis=0)
        optwv[mask_mid] = g * WIN[max_idx]

    # --- vectorized Nadaraya-Watson kernel regression ---
    sigma = optwv / g                            # (L,)
    t_diff = t[:, np.newaxis] - t[np.newaxis, :] # (L, L)

    if WinFunc == 'Boxcar':
        a = 12**0.5 * sigma                      # (L,)
        Z = np.where(np.abs(t_diff) <= a[np.newaxis, :] / 2,
                     1 / a[np.newaxis, :], 0)
    elif WinFunc == 'Laplace':
        Z = (1 / 2**0.5 / sigma[np.newaxis, :]
             * np.exp(-(2**0.5) / sigma[np.newaxis, :] * np.abs(t_diff)))
    elif WinFunc == 'Cauchy':
        Z = 1 / (np.pi * sigma[np.newaxis, :]
                 * (1 + (t_diff / sigma[np.newaxis, :])**2))
    else:  # Gauss
        Z = (1 / (2 * np.pi)**2 / sigma[np.newaxis, :]
             * np.exp(-t_diff**2 / 2 / sigma[np.newaxis, :]**2))

    optwp = np.sum(optwv[np.newaxis, :] * Z, axis=1) / np.sum(Z, axis=1)

    # --- vectorized balloon estimator ---
    idx = y_hist.nonzero()
    y_hist_nz = y_hist[idx]
    t_nz = t[idx]
    t_diff_nz = t[:, np.newaxis] - t_nz[np.newaxis, :]  # (L, nnz)
    G = (1 / (2 * np.pi)**2 / optwp[:, np.newaxis]
         * np.exp(-t_diff_nz**2 / 2 / optwp[:, np.newaxis]**2))
    yv = np.sum(y_hist_nz[np.newaxis, :] * dt * G, axis=1)
    yv = yv * N / np.sum(yv * dt)

    # cost function of estimated kernel
    cg = yv**2 - 2 * yv * y_hist + 2 / (2 * np.pi)**0.5 / optwp * y_hist
    Cg = np.sum(cg * dt)

    return Cg, yv, optwp


_rfreq_cache = {}


def _get_rfreq(n):
    if n not in _rfreq_cache:
        _rfreq_cache[n] = np.fft.rfftfreq(n)
    return _rfreq_cache[n]


def fftkernel(x, w):
    # forward padded transform
    L = x.size
    Lmax = L + 3 * w
    n = int(2 ** np.ceil(np.log2(Lmax)))
    X = np.fft.rfft(x, n)

    # generate kernel domain (cached)
    f = _get_rfreq(n)

    # evaluate kernel
    K = np.exp(-0.5 * (w * 2 * np.pi * f) ** 2)

    # convolve and transform back from frequency domain
    y = np.fft.irfft(X * K, n)
    y = y[0:L]

    return y


def fftkernelWin(x, w, WinFunc):
    # forward padded transform
    L = x.size
    Lmax = L + 3 * w
    n = int(2 ** np.ceil(np.log2(Lmax)))
    X = np.fft.rfft(x, n)

    # generate kernel domain (cached)
    f = _get_rfreq(n)
    t = 2 * np.pi * f

    # determine window function - evaluate kernel
    if WinFunc == 'Boxcar':
        a = 12**0.5 * w
        K = np.zeros(len(t))
        K[0] = 1
        K[1:] = 2 * np.sin(a * t[1:] / 2) / (a * t[1:])
    elif WinFunc == 'Laplace':
        K = 1 / (1 + (w * 2 * np.pi * f)**2 / 2)
    elif WinFunc == 'Cauchy':
        K = np.exp(-w * np.abs(2 * np.pi * f))
    else:  # WinFunc == 'Gauss'
        K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)

    # convolve and transform back from frequency domain
    y = np.fft.irfft(X * K, n)
    y = y[0:L]

    return y


def Gauss(x, w):
    y = 1 / (2 * np.pi)**2 / w * np.exp(-x**2 / 2 / w**2)
    return y


def Laplace(x, w):
    y = 1 / 2**0.5 / w * np.exp(-(2**0.5) / w * np.abs(x))
    return y


def Cauchy(x, w):
    y = 1 / (np.pi * w * (1 + (x / w)**2))
    return y


def Boxcar(x, w):
    a = 12**0.5 * w
    y = 1 / a
    y[np.abs(x) > a / 2] = 0
    return y


def logexp(x):
    y = np.zeros(x.shape)
    y[x < 1e2] = np.log(1+np.exp(x[x < 1e2]))
    y[x >= 1e2] = x[x >= 1e2]
    return y


def ilogexp(x):
    y = np.zeros(x.shape)
    y[x < 1e2] = np.log(np.exp(x[x < 1e2]) - 1)
    y[x >= 1e2] = x[x >= 1e2]
    return y
    