import numpy as np

try:
    from scipy.fft import rfft, irfft, rfftfreq
except ImportError:
    from numpy.fft import rfft, irfft, rfftfreq


def sskernel(x, tin=None, W=None, nbs=1000):
    """
    Generates a kernel density estimate with globally-optimized bandwidth.

    The optimal bandwidth is obtained as a minimizer of the formula, sum_{i,j}
    \int k(x - x_i) k(x - x_j) dx  -  2 sum_{i~=j} k(x_i - x_j), where k(x) is
    the kernel function.


    Parameters
    ----------
    x : array_like
        The one-dimensional samples drawn from the underlying density
    tin : array_like, optional
        The values where the density estimate is to be evaluated in generating
        the output 'y'.
    W : array_like, optional
        The kernel bandwidths to use in optimization. Should not be chosen
        smaller than the sampling resolution of 'x'.
    nbs : int, optional
        The number of bootstrap samples to use in estimating the [0.05, 0.95]
        confidence interval of the output 'y'

    Returns
    -------
    y : array_like
        The estimated density, evaluated at points t / tin.
    t : array_like
        The points where the density estimate 'y' is evaluated.
    optw : double
        The optimal global kernel bandwidth.
    W : array_like
        The kernel bandwidths evaluated during optimization.
    C : array_like
        The cost functions associated with the bandwidths 'W'.
    confb95 : array_like
        The 5% and 95% confidence interval of the kernel density estimate 'y'.
        Has dimensions 2 x len(y). confb95[0,:] corresponds to the 5% interval,
        and confb95[1,:] corresponds to the 95% interval.
    yb : array_like
        The bootstrap samples used in estimating confb95. Each row corresponds
        to one bootstrap sample.

    See Also
    --------
    sshist, ssvkernel

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
        T = np.max(tin) - np.min(tin)
        x_ab = x[(x >= min(tin)) & (x <= max(tin))]
        dx = np.sort(np.diff(np.sort(x_ab)))
        dt_samp = dx[np.nonzero(dx)][0]
        if dt_samp > min(np.diff(tin)):
            t = np.linspace(min(tin), max(tin), int(min(np.ceil(T / dt_samp), 1e3)))
        else:
            t = tin

    # calculate delta t
    dt = np.min(np.diff(t))

    # create the finest histogram
    thist = np.concatenate((t, (t[-1]+dt)[np.newaxis]))
    y_hist = np.histogram(x_ab, thist-dt/2)[0]
    N = np.sum(y_hist).astype(float)
    y_hist = y_hist / N / dt

    # global search if input 'W' is defined
    if W is not None:
        C = np.zeros((1, len(W)))
        C_min = np.inf
        for k, w in enumerate(W):
            C[k], yh = CostFunction(y_hist, N, w, dt)
            if(C[k] < C_min):
                C_min = C[k]
                optw = w
                y = yh
    else:  # optimized search using golden section
        k = 0
        C = np.zeros((20, 1))
        W = np.zeros((20, 1))
        Wmin = 2*dt
        Wmax = (np.max(x) - np.min(x))
        tol = 1e-5
        phi = (5**0.5 + 1) / 2
        a = ilogexp(Wmin)
        b = ilogexp(Wmax)
        c1 = (phi - 1) * a + (2 - phi) * b
        c2 = (2 - phi) * a + (phi - 1) * b
        f1, dummy = CostFunction(y_hist, N, logexp(c1), dt)
        f2, dummy = CostFunction(y_hist, N, logexp(c2), dt)
        while (np.abs(b-a) > tol * (np.abs(c1) + np.abs(c2))) & (k < 20):
            if f1 < f2:
                b = c2
                c2 = c1
                c1 = (phi - 1) * a + (2 - phi) * b
                f2 = f1
                f1, yh1 = CostFunction(y_hist, N, logexp(c1), dt)
                W[k] = logexp(c1)
                C[k] = f1
                optw = logexp(c1)
                y = yh1 / np.sum(yh1 * dt)
            else:
                a = c1
                c1 = c2
                c2 = (2 - phi) * a + (phi - 1) * b
                f1 = f2
                f2, yh2 = CostFunction(y_hist, N, logexp(c2), dt)
                W[k] = logexp(c2)
                C[k] = f2
                optw = logexp(c2)
                y = yh2 / np.sum(yh2 * dt)

            # increment iteration counter
            k = k + 1

        # discard unused entries in gs, C
        C = C[0:k]
        W = W[0:k]

    # estimate confidence intervals by bootstrapping
    nbs = np.asarray(nbs)
    L = len(t)
    thist = np.concatenate((t, (t[-1]+dt)[np.newaxis]))
    bins = thist - dt / 2

    # generate all bootstrap histograms
    y_all = np.zeros((nbs, L))
    for i in range(nbs):
        idx = np.random.randint(0, len(x_ab), len(x_ab))
        xb = x_ab[idx]
        y_all[i, :] = np.histogram(xb, bins)[0] / dt / N

    # batched FFT convolution (single 2D rfft instead of nbs individual calls)
    w = optw / dt
    n = int(2 ** np.ceil(np.log2(L + 3 * w)))
    Y_all = rfft(y_all, n, axis=1)
    if n not in _rfreq_cache:
        _rfreq_cache[n] = rfftfreq(n)
    f = _rfreq_cache[n]
    K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)
    yb_conv = irfft(Y_all * K[np.newaxis, :], n, axis=1)[:, :L]

    # normalize and interpolate
    norms = np.sum(yb_conv * dt, axis=1, keepdims=True)
    yb_conv = yb_conv / norms
    yb = np.zeros((nbs, len(tin)))
    for i in range(nbs):
        yb[i, :] = np.interp(tin, t, yb_conv[i, :])
    ybsort = np.sort(yb, axis=0)
    y95b = ybsort[int(np.floor(0.05 * nbs)), :]
    y95u = ybsort[int(np.floor(0.95 * nbs)), :]
    confb95 = np.concatenate((y95b[np.newaxis], y95u[np.newaxis]), axis=0)

    # return outputs
    y = np.interp(tin, t, y)
    t = tin

    return y, t, optw, W, C, confb95, yb


def CostFunction(y_hist, N, w, dt):

    # build normal smoothing kernel
    yh = fftkernel(y_hist, w / dt)

    # formula for density
    C = np.sum(yh**2) * dt - 2 * np.sum(yh * y_hist) * dt + 2 \
        / (2 * np.pi)**0.5 / w / N
    C = C * N**2

    return C, yh


_rfreq_cache = {}


def fftkernel(x, w):

    L = x.size
    Lmax = L + 3 * w
    n = int(2 ** np.ceil(np.log2(Lmax)))

    X = rfft(x, n)

    if n not in _rfreq_cache:
        _rfreq_cache[n] = rfftfreq(n)
    f = _rfreq_cache[n]

    K = np.exp(-0.5 * (w * 2 * np.pi * f) ** 2)

    y = irfft(X * K, n)

    y = y[0:L]

    return y


def logexp(x):
    if x < 1e2:
        y = np.log(1 + np.exp(x))
    else:
        y = x
    return y


def ilogexp(x):
    # ilogexp = log(exp(x)-1);
    if x < 1e2:
        y = np.log(np.exp(x) - 1)
    else:
        y = x
    return y
