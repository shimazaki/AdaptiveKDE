# AdaptiveKDE Performance Optimization Report

Branch: `opt-experiments`
Date: February 2026
Platform: carnot (2x Xeon 6258R, 754 GB RAM), Python 3.11.14, NumPy 2.4.2

## 1. Summary

All three functions in AdaptiveKDE were systematically optimized using NumPy
vectorization, scipy.fft, and numba JIT compilation. Every optimization preserves
numerical correctness, verified by 18 golden reference checks at `rtol=1e-10`.

| Function | Baseline (s) | Optimized (s) | Speedup |
|---|---|---|---|
| `sshist` | 0.194 | 0.004 | **44x** |
| `sskernel` | 0.024 | 0.010 | **2.4x** |
| `ssvkernel` | 0.954 | 0.070 | **13.6x** |
| **Total** | **1.172** | **0.085** | **13.8x** |

CPU time (median of 20 runs, 3 warmup discarded) on 107-point Old Faithful data.

Optional dependencies: `scipy` (faster FFT backend), `numba` (JIT for sshist).
Falls back to pure NumPy when not installed (12.4x speedup without them).

## 2. Benchmark Setup

- **Metric**: CPU time via `time.process_time()` (excludes I/O wait, measures
  actual computation)
- **Runs**: 3 warmup (discarded) + 20 timed runs; median reported
- **Datasets**: Old Faithful eruption durations (n=107) and synthetic bimodal
  mixture (n=1000)
- **Verification**: `python tests/run_tests.py` — 18 golden checks at
  `rtol=1e-10` must all pass after every change
- **Harness**: `tests/run_opt_experiments.py` records results to JSON;
  `--report` prints comparison table

## 3. Optimization Details

Each optimization was applied cumulatively (each step builds on all previous
steps). The tables below show incremental and cumulative speedups.

---

### 3.1 D1: Replace `sum()`/`min()` with `np.sum()`/`np.min()`

**Target**: all three functions
**Incremental speedup**: 0.97x (no gain)

Replaced Python built-in `sum()` and `min()` with NumPy equivalents across all
files. For the small arrays typical in this codebase (L ~ 327), the overhead of
dispatching to NumPy slightly outweighs the benefit. This change was kept for
consistency but provides no measurable improvement.

---

### 3.2 A1: Replace `np.histogram` with `np.searchsorted` in `sshist`

**Target**: `sshist`
**Incremental speedup**: 1.67x for sshist

The original `sshist` called `np.histogram(x, edges)` inside a double loop
(~162 bin counts x 30 shifts = 4860 calls for n=107). Each `np.histogram` call
internally sorts the data and performs a binary search.

**Optimization**: Sort `x` once at the start, then use `np.searchsorted` +
`np.diff` for bin counting:

```python
# Original (per iteration):
counts = np.histogram(x, edges)[0]

# Optimized (sort once, reuse):
x_sorted = np.sort(x)
# ... inside loop:
counts = np.diff(np.searchsorted(x_sorted, edges))
```

`np.searchsorted` on pre-sorted data is O(n_edges * log n), avoiding the
repeated O(n log n) sort inside `np.histogram`.

---

### 3.3 B1 + C2 + C5: Cache frequency vectors and pre-allocate arrays

**Target**: `sskernel` (B1) and `ssvkernel` (C2, C5)
**Incremental speedup**: 1.17x total

Three related micro-optimizations applied together:

- **B1/C2 — Cache `rfftfreq`**: The FFT convolution functions call
  `np.fft.rfftfreq(n)` to generate frequency vectors. Since padded FFT sizes
  are reused across iterations (only ~3 distinct sizes), caching by `n`
  eliminates redundant allocations:

  ```python
  _rfreq_cache = {}

  def _get_rfreq(n):
      if n not in _rfreq_cache:
          _rfreq_cache[n] = np.fft.rfftfreq(n)
      return _rfreq_cache[n]
  ```

- **C5 — Pre-allocate `C_local`**: In the `ssvkernel` M x M loop, a temporary
  array `C_local` of shape `(M, L)` was allocated on every iteration. Moving the
  allocation outside the loop avoids ~80 repeated allocations.

---

### 3.4 D2: Replace `fft`/`ifft` with `rfft`/`irfft`

**Target**: `sskernel` and `ssvkernel` (all `fftkernel` / `fftkernelWin` calls)
**Incremental speedup**: 1.12x total (sskernel 1.11x, ssvkernel 1.14x)

Since all input signals are real-valued, `np.fft.rfft` computes only the
non-negative frequencies (n/2 + 1 coefficients instead of n), roughly halving
both the FFT computation and the size of the frequency-domain kernel:

```python
# Original:
X = np.fft.fft(x, n)
f = np.fft.fftfreq(n)
K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)
y = np.fft.ifft(X * K, n).real

# Optimized:
X = np.fft.rfft(x, n)         # half the output size
f = np.fft.rfftfreq(n)        # half the frequency vector
K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)
y = np.fft.irfft(X * K, n)    # no .real needed
```

---

### 3.5 C1: Batch FFTs in the M x M loop of `ssvkernel`

**Target**: `ssvkernel` (local bandwidth optimization loop)
**Incremental speedup**: 1.48x for ssvkernel

This is the core computational bottleneck of `ssvkernel`. The original code had
a nested loop: for each of M=80 outer window sizes, it convolved all M=80 cost
function rows with a smoothing kernel — totaling M x M = 6400 individual FFT
calls.

**Key insight**: The padded FFT size `n` depends on the window width `w`, but
only ~3 distinct sizes occur across all 80 windows. By grouping iterations by
FFT size, the forward FFT of the cost matrix `c` (shape M x L) can be computed
just once per group, then each inner iteration only needs a kernel multiply and
inverse FFT.

```python
# Group outer iterations by padded FFT size
fft_groups = {}
for i in range(M):
    w = W[i] / dt
    n_fft = int(2 ** np.ceil(np.log2(L + 3 * w)))
    fft_groups.setdefault(n_fft, []).append((i, w))

for n_fft, group in fft_groups.items():
    # Forward FFT of ALL M cost rows at this padded size — done ONCE
    C_fft = np.fft.rfft(c, n_fft, axis=1)    # (M, n_fft//2+1)
    f = _get_rfreq(n_fft)

    for i, w in group:
        K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)  # kernel for this w
        # Batched IFFT: apply kernel to all M rows at once
        C_local = np.fft.irfft(C_fft * K[np.newaxis, :], n_fft, axis=1)[:, :L]
        optws[i, :] = W[np.argmin(C_local, axis=0)]
```

This reduces ~6400 individual FFTs to ~3 batched forward FFTs + ~80 batched
inverse FFTs.

---

### 3.6 C3: Vectorize `CostFunction` with NumPy broadcasting

**Target**: `ssvkernel` `CostFunction`
**Incremental speedup**: 2.37x for ssvkernel

`CostFunction` is called ~27 times during the golden section search. The
original implementation had three Python `for k in range(L)` loops (L ~ 327):

**Loop 1 — Bandwidth selection (`optwv`)**: For each grid point, find the
optimal bandwidth from a lookup table using conditional logic. Vectorized using
masked array operations:

```python
# Original: for k in range(L): ... if/elif/else with np.nonzero ...

# Vectorized:
gs_all = optws / WIN[:, np.newaxis]       # (M, L) ratio matrix
gs_max = np.max(gs_all, axis=0)           # (L,)
gs_min = np.min(gs_all, axis=0)           # (L,)

optwv = np.full(L, WIN_max)               # default case
optwv[g > gs_max] = WIN_min               # high-g case
mask_mid = ~(g > gs_max) & (g >= gs_min)  # interpolation case
if np.any(mask_mid):
    ge_mask = gs_all[:, mask_mid] >= g
    max_idx = np.max(np.where(ge_mask, row_idx, -1), axis=0)
    optwv[mask_mid] = g * WIN[max_idx]
```

**Loop 2 — Nadaraya-Watson kernel regression**: Computes a weighted average of
bandwidths using a kernel function. The original evaluates the kernel L times,
each producing a vector of length L. Vectorized into a single L x L matrix
operation:

```python
# Original: for k in range(L): Z = Gauss(t[k]-t, sigma); optwp[k] = ...

# Vectorized: compute full L×L pairwise kernel matrix at once
t_diff = t[:, np.newaxis] - t[np.newaxis, :]   # (L, L) distance matrix
Z = (1 / (2*np.pi)**2 / sigma[np.newaxis, :]
     * np.exp(-t_diff**2 / 2 / sigma[np.newaxis, :]**2))
optwp = np.sum(optwv * Z, axis=1) / np.sum(Z, axis=1)
```

**Loop 3 — Balloon estimator**: Similar pattern but operates on the L x nnz
submatrix (only nonzero histogram bins), which is typically much smaller than
L x L:

```python
# Original: for k in range(L): yv[k] = sum(y_hist_nz * dt * Gauss(...))

# Vectorized:
t_diff_nz = t[:, np.newaxis] - t_nz[np.newaxis, :]   # (L, nnz)
G = (1 / (2*np.pi)**2 / optwp[:, np.newaxis]
     * np.exp(-t_diff_nz**2 / 2 / optwp[:, np.newaxis]**2))
yv = np.sum(y_hist_nz * dt * G, axis=1)
```

For L=327 with the default Gauss kernel, the L x L matrix is ~850 KB — well
within cache. NumPy's vectorized `exp` and matrix operations are dramatically
faster than 327 individual Python-level function calls.

---

### 3.7 C4: Vectorize bootstrap inner loop in `ssvkernel`

**Target**: `ssvkernel` bootstrap confidence intervals
**Incremental speedup**: 2.81x for ssvkernel

The bootstrap loop runs `nbs` iterations (default 50–100), and within each
iteration, the original code had a `for k in range(L)` loop computing the
kernel density at each grid point. This is the same L x nnz pattern as the
balloon estimator in CostFunction:

```python
# Original:
for k in range(L):
    yb_buf[k] = np.sum(y_histb_nz * Gauss(t[k] - t_nz, optw[k])) / Nb

# Vectorized:
t_diff_nz = t[:, np.newaxis] - t_nz[np.newaxis, :]
G = (inv_2pi2 / optw[:, np.newaxis]
     * np.exp(-t_diff_nz**2 / optw_sq2[:, np.newaxis]))
yb_buf = np.sum(y_histb_nz * G, axis=1) / Nb
```

Additionally, loop-invariant computations were hoisted outside the bootstrap
loop:
- `thist` and `bins` (histogram bin edges) — constant across iterations
- `inv_2pi2 = 1 / (2*pi)^2` and `optw_sq2 = 2 * optw**2` — pre-computed once

---

### 3.8 B2: Batch bootstrap FFTs in `sskernel`

**Target**: `sskernel` bootstrap confidence intervals
**Incremental speedup**: 1.80x for sskernel

The `sskernel` bootstrap generates `nbs` (default 200) bootstrap histograms,
each convolved with the same Gaussian kernel at the optimal bandwidth `optw`.
The original code called `fftkernel` individually for each bootstrap sample.

Since all samples share the same kernel, we can batch the FFT convolution into
a single 2D operation:

```python
# Original: nbs individual FFT calls
for i in range(nbs):
    ...
    yb_buf = fftkernel(y_histb, optw / dt)

# Optimized: generate all histograms first, then single batched FFT
y_all = np.zeros((nbs, L))
for i in range(nbs):
    ...
    y_all[i, :] = np.histogram(xb, bins)[0] / dt / N

# Single 2D rfft → kernel multiply → 2D irfft
Y_all = np.fft.rfft(y_all, n, axis=1)       # (nbs, n//2+1)
K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)   # (n//2+1,) — same for all
yb_conv = np.fft.irfft(Y_all * K[np.newaxis, :], n, axis=1)[:, :L]
```

This replaces 200 individual FFT+IFFT pairs with a single batched 2D FFT+IFFT,
which NumPy/FFTPACK handles much more efficiently.

---

### 3.9 E2: Pre-compute pairwise distances for `CostFunction`

**Target**: `ssvkernel` `CostFunction`
**Incremental speedup**: 1.07x for ssvkernel

`CostFunction` is called ~27 times during the golden section search, but several
quantities are invariant across calls:

| Quantity | Shape | Recomputed | Cost |
|---|---|---|---|
| `t_diff = t[:,None] - t[None,:]` | (L, L) | ~27 times | O(L^2) |
| `t_diff_nz = t[:,None] - t_nz[None,:]` | (L, nnz) | ~27 times | O(L * nnz) |
| `gs_all = optws / WIN[:,None]` | (M, L) | ~27 times | O(M * L) |
| `gs_max`, `gs_min` | (L,) | ~27 times | O(M * L) |
| `y_hist_nz`, `WIN_min`, `WIN_max` | scalars/vectors | ~27 times | O(L) |

All of these are computed once before the golden section loop and passed to
`CostFunction` via a `precomp` dictionary. The function signature gains an
optional `precomp` parameter (backward compatible: defaults to `None`, in which
case quantities are computed internally).

---

### 3.10 A3: Batch shift loop in `sshist` with vectorized `searchsorted`

**Target**: `sshist`
**Incremental speedup**: 8.43x for sshist

The remaining bottleneck was the inner loop over `SN=30` shift positions in
`sshist`. For each shift, the code computed bin edges via `np.linspace` and
counted events via `np.searchsorted`. This was vectorized by constructing all
30 sets of edges simultaneously via broadcasting, then performing a single
flattened `searchsorted` call:

```python
# Original: loop over SN=30 shifts
for p, sh in enumerate(shift):
    edges = np.linspace(x_min + sh - D/2, x_max + sh - D/2, n+1)
    counts = np.diff(np.searchsorted(x_sorted, edges))
    ...

# Vectorized: build (SN, n+1) edge matrix, single searchsorted call
lo = x_min + shift - D / 2                          # (SN,)
hi = x_max + shift - D / 2                          # (SN,)
frac = np.linspace(0, 1, n + 1)                     # (n+1,)
all_edges = lo[:, None] + frac[None, :] * (hi - lo)[:, None]  # (SN, n+1)

ss = np.searchsorted(x_sorted, all_edges.ravel())   # single call
counts = np.diff(ss.reshape(SN, n + 1), axis=1)     # (SN, n)
k = counts.mean(axis=1)                             # (SN,)
v = np.sum((counts - k[:, None])**2, axis=1) / n    # (SN,)
Cs[i, :] = (2 * k - v) / D**2
```

The key insight is that `np.searchsorted` operates on a single sorted array and
can process an arbitrarily large query array in one call. By flattening the
(SN, n+1) edge matrix into a 1D array, we replace 30 individual `searchsorted`
calls with one, and the subsequent `np.diff` and statistics are fully vectorized
along the shift axis.

---

## 4. Cumulative Results

### 4.1 n=107 (Old Faithful)

| # | Optimization | sshist (s) | sskernel (s) | ssvkernel (s) | Total (s) | vs Baseline |
|---|---|---|---|---|---|---|
| 0 | baseline | 0.194 | 0.024 | 0.954 | 1.172 | 1.00x |
| 1 | D1: sum→np.sum | 0.204 | 0.025 | 0.979 | 1.208 | 0.97x |
| 2 | A1: searchsorted | 0.122 | 0.025 | 0.971 | 1.118 | 1.05x |
| 3 | B1+C2+C5: cache+prealloc | 0.116 | 0.020 | 0.822 | 0.958 | 1.22x |
| 4 | D2: rfft/irfft | 0.121 | 0.018 | 0.718 | 0.857 | 1.37x |
| 5 | C1: batch FFTs | 0.115 | 0.018 | 0.485 | 0.617 | 1.90x |
| 6 | C3: vectorize CostFunction | 0.116 | 0.018 | 0.205 | 0.339 | 3.46x |
| 7 | C4: vectorize bootstrap | 0.123 | 0.018 | 0.073 | 0.214 | 5.48x |
| 8 | B2: batch bootstrap FFTs | 0.117 | 0.010 | 0.072 | 0.199 | 5.88x |
| 9 | E2: precomp pairwise | 0.118 | 0.010 | 0.068 | 0.197 | 5.96x |
| 10 | A3: batch shift searchsorted | 0.014 | 0.010 | 0.070 | 0.095 | 12.37x |
| 11 | B3: scipy.fft | 0.015 | 0.010 | 0.069 | 0.094 | 12.44x |
| 12 | A2: numba JIT sshist | 0.004 | 0.010 | 0.071 | 0.085 | 13.73x |
| 13 | C6: numba ssvkernel (no gain) | 0.004 | 0.010 | 0.070 | 0.085 | 13.80x |
| | **final-combined** | **0.004** | **0.010** | **0.070** | **0.085** | **13.82x** |

### 4.2 n=1000 (Synthetic bimodal)

| # | Optimization | sshist (s) | sskernel (s) | ssvkernel (s) | Total (s) | vs Baseline |
|---|---|---|---|---|---|---|
| 0 | baseline | 0.732 | 0.047 | 2.574 | 3.353 | 1.00x |
| 10 | numpy-only final | 0.119 | 0.023 | 0.511 | 0.652 | 5.14x |
| | **final-combined** | **0.112** | **0.022** | **0.513** | **0.647** | **5.18x** |

---

## 5. Incremental Speedup per Step

Each row shows the speedup relative to the *previous* step, isolating the
contribution of that single optimization.

| # | Optimization | Target function | Before → After (s) | Step speedup |
|---|---|---|---|---|
| 1 | D1: sum→np.sum | all | 1.172 → 1.208 | **0.97x** (no gain) |
| 2 | A1: searchsorted | sshist | 0.204 → 0.122 | **1.67x** |
| 3 | B1+C2+C5: cache+prealloc | sskernel+ssvkernel | 0.025→0.020, 0.971→0.822 | **1.17x** |
| 4 | D2: rfft/irfft | sskernel+ssvkernel | 0.020→0.018, 0.822→0.718 | **1.12x** |
| 5 | C1: batch FFTs | ssvkernel | 0.718 → 0.485 | **1.48x** |
| 6 | C3: vectorize CostFunction | ssvkernel | 0.485 → 0.205 | **2.37x** |
| 7 | C4: vectorize bootstrap | ssvkernel | 0.205 → 0.073 | **2.81x** |
| 8 | B2: batch bootstrap FFTs | sskernel | 0.018 → 0.010 | **1.80x** |
| 9 | E2: precomp pairwise | ssvkernel | 0.073 → 0.068 | **1.07x** |
| 10 | A3: batch shift searchsorted | sshist | 0.118 → 0.014 | **8.43x** |
| 11 | B3: scipy.fft | sskernel+ssvkernel | 0.095 → 0.094 | **1.01x** (marginal) |
| 12 | A2: numba JIT sshist | sshist | 0.014 → 0.004 | **3.27x** |
| 13 | C6: numba ssvkernel | ssvkernel | 0.070 → 0.070 | **1.00x** (no gain) |

The three largest single-step wins:
1. **A3** (8.43x) — eliminated the 30-iteration inner Python loop in sshist
2. **A2** (3.27x) — numba JIT compiled the entire sshist triple loop
3. **C4** (2.81x) — replaced `for k in range(L)` with L x nnz broadcasting in bootstrap

---

## 6. Per-Function Breakdown

### 6.1 `sshist`: 0.194s → 0.004s (44x)

The original bottleneck was a double loop: ~162 bin counts x 30 shifts = ~4860
iterations, each calling `np.histogram`. Three optimizations addressed this:

1. **A1** (1.67x): Replace `np.histogram` with `np.searchsorted` on pre-sorted
   data, eliminating redundant sorting
2. **A3** (8.43x): Vectorize the inner 30-shift loop by constructing all edge
   arrays via broadcasting and calling `searchsorted` once per bin count
3. **A2** (3.27x): Numba JIT compiles the entire triple loop (bins x shifts x
   searchsorted) into native machine code, eliminating all Python overhead

With numba, the outer loop over bin counts is also compiled. The entire cost
function computation runs as a single native function call.

### 6.2 `sskernel`: 0.024s → 0.010s (2.4x)

`sskernel` was already fast (only ~2% of total time). Improvements came from:

1. **D2** (1.11x): `rfft`/`irfft` instead of `fft`/`ifft`
2. **B1** (included in 1.17x): Cached frequency vectors
3. **B2** (1.80x): Batched 200 bootstrap FFTs into single 2D `rfft`/`irfft`

### 6.3 `ssvkernel`: 0.954s → 0.070s (13.6x)

`ssvkernel` was the dominant cost (81% of baseline time). The optimizations
targeted three computational phases:

**Phase 1 — Local bandwidth optimization (M x M FFT loop):**
- C2+C5 (cache + prealloc): 1.18x
- D2 (rfft): 1.14x
- C1 (batch by FFT size): 1.48x

**Phase 2 — Golden section search (CostFunction, called ~27 times):**
- C3 (vectorize 3 loops): 2.37x
- E2 (precompute invariants): 1.07x

**Phase 3 — Bootstrap confidence intervals (50 iterations):**
- C4 (vectorize inner loop): 2.81x

---

## 7. Optional Dependency Results

### 7.1 B3: `scipy.fft` — marginal improvement (1.01x)

**Target**: `sskernel` and `ssvkernel` (all FFT calls)
**Incremental speedup**: 1.01x total

`scipy.fft` is ~23x faster per-call than `numpy.fft` in microbenchmarks.
However, after batching optimizations (C1, B2), there are very few individual
FFT calls remaining. The per-call speedup has almost no impact on total runtime.

Both files use a fallback import pattern:

```python
try:
    from scipy.fft import rfft, irfft, rfftfreq
except ImportError:
    from numpy.fft import rfft, irfft, rfftfreq
```

### 7.2 A2: Numba JIT for `sshist` — strong improvement (3.27x)

**Target**: `sshist`
**Incremental speedup**: 3.27x for sshist (0.014s → 0.004s)

After vectorizing the shift loop (A3), the remaining overhead is the outer loop
over ~162 bin counts, plus Python-level dispatch of NumPy operations per
iteration. Numba JIT compiles the entire triple loop (bins x shifts x
searchsorted) into a single native function:

```python
@njit(cache=True)
def _cost(x_sorted, x_min, x_max, N_MIN, N_MAX, SN):
    for i in range(n_range):         # bin counts
        for p in range(SN):          # shifts
            prev_idx = np.searchsorted(x_sorted, base)
            for b in range(1, n+1):  # bins
                edge = base + span * b / n
                cur_idx = np.searchsorted(x_sorted, edge)
                count = cur_idx - prev_idx
                ...
```

Key implementation details:
- Edge computation uses `base + span * b / n` to match `np.linspace` behavior
  (avoids floating-point drift that would break golden tests at `rtol=1e-10`)
- Uses `cache=True` for persistent compilation — first call incurs ~0.7s JIT
  overhead, subsequent calls use the cached native code
- Falls back to vectorized NumPy when numba is not installed

### 7.3 C6: Numba JIT for `ssvkernel` CostFunction — no improvement

**Target**: `ssvkernel` `CostFunction` and bootstrap
**Incremental speedup**: 1.00x (no gain, code reverted)

Numba JIT was tested for the CostFunction (replacing L x L broadcasting with
explicit double loops) and bootstrap inner kernel. Result: identical performance
for n=107, slightly *worse* for n=1000.

**Why numba didn't help here**: The vectorized NumPy broadcasting in C3/C4
already operates at near-optimal speed for these array sizes:
- L x L matrix (327 x 327 = ~850 KB) fits in L2 cache
- NumPy's BLAS-backed `exp` and matrix multiply operate at near-native speed
- Numba's overhead from the JIT boundary and scalar `exp()` calls negated any
  loop-fusion advantage
- For larger arrays (L > 1000), numba might help by avoiding temporary array
  allocation, but the current workloads don't benefit

## 8. Optimizations Not Applied

| ID | Optimization | Reason |
|---|---|---|
| E1 | Multiprocessing bootstrap | Bootstrap is only ~68ms after C4; process creation overhead would negate gains for 50 iterations at ~1ms each |
| D1 | sum→np.sum | Applied but showed no benefit (0.97x); kept for consistency |

---

## 9. Algorithmic Principles

The optimizations applied here follow several general principles:

1. **Eliminate redundant computation**: Sort data once (A1), cache frequency
   vectors (B1/C2), pre-compute distance matrices (E2)

2. **Replace Python loops with NumPy broadcasting**: The largest wins (C3, C4,
   A3) all come from converting `for k in range(L)` loops into matrix operations.
   NumPy's C-level vectorized operations are 100-1000x faster than equivalent
   Python loops for arrays of size 100-1000.

3. **Batch FFT operations**: Individual FFT calls have Python-level overhead.
   Batching into 2D FFTs (C1, B2) amortizes this overhead and enables better
   memory access patterns.

4. **Use appropriate FFT variants**: `rfft`/`irfft` for real-valued data (D2)
   halves the frequency-domain computation with zero algorithmic change.

5. **Group by shared structure**: The M x M FFT loop has only ~3 distinct padded
   sizes. Grouping by size (C1) converts O(M^2) individual FFTs into
   O(n_groups) batched FFTs.

6. **JIT compilation for irreducible loops**: When a loop cannot be vectorized
   (sshist outer loop over variable bin counts), numba `@njit` compiles it to
   native code. But for already-vectorized code (ssvkernel), numba provides
   no additional benefit — vectorized NumPy is already near-optimal.

7. **Diminishing returns from FFT backends**: `scipy.fft` is 23x faster
   per-call, but after batching reduces the call count from thousands to
   single-digits, the per-call speedup becomes irrelevant.

---

## 10. Computational Complexity

### The grid resolution parameter L

All three functions operate on an internal evaluation grid of `L` points, not
directly on the raw samples. `L` is determined by data resolution:

```python
L = int(min(np.ceil(T / dt_samp), 1e3))
```

where `T = max(x) - min(x)` is the data range and `dt_samp` is the smallest
nonzero gap between consecutive sorted samples. The hardcoded cap of `1e3`
limits `L` to at most 1000 grid points.

**L depends on data resolution, not on sample count n.** For example:

| Dataset | n | dt_samp | T / dt_samp | L |
|---|---|---|---|---|
| Old Faithful (107 pts) | 107 | 0.01 | 326 | 327 |
| 10,000 pts rounded to 2 decimals | 10,000 | 0.01 | ~326 | 327 |
| 10,000 pts at full precision | 10,000 | ~3e-4 | ~10,000 | **1000** (capped) |

For high-precision data with many samples, `L` saturates at 1000. This means
the O(L^2) operations in CostFunction become a constant cost, and overall
scaling becomes linear in n.

### Complexity by function

#### `sshist`

| Phase | Operation | Complexity |
|---|---|---|
| Sort data | `np.sort(x)` | O(n log n) |
| Outer loop | N_MAX bin counts | N_MAX = min(T / 2dx, 500) |
| Inner (vectorized) | SN × n_bins searchsorted queries on n points | O(SN × n_bins × log n) |
| **Total** | sum over bin counts | **O(N_MAX^2 × log n + n log n)** |

N_MAX is capped at `max(N)` = 500 by default. For well-resolved data with
small dx, N_MAX can grow proportional to n until hitting the cap:

- **n < ~1000**: O(n^2 log n) — N_MAX grows with n
- **n > ~1000**: O(n log n) — N_MAX capped, sort dominates

#### `sskernel`

| Phase | Operation | Complexity |
|---|---|---|
| Setup (sort, histogram) | | O(n log n) |
| Golden section search | ~20 iterations × fftkernel | O(20 × L log L) |
| Bootstrap (dominant) | nbs × (histogram + FFT) | O(nbs × (n + L log L)) |
| **Total** | | **O(nbs × n + nbs × L log L)** |

The bootstrap histogram generation O(nbs × n) dominates for large n. With
L capped at 1000 and the default nbs = 1000:

- **n < 1000**: O(nbs × n log n) — L ≈ n, FFT phase matters
- **n > 1000**: O(nbs × n) — linear in n

#### `ssvkernel`

| Phase | Operation | Complexity |
|---|---|---|
| Local cost (M FFTs) | M = 80 fftkernel calls | O(M × L log L) |
| M×M bandwidth selection | batched FFTs by group | O(M^2 × L log L) |
| CostFunction (×~27) | L×L Nadaraya-Watson + L×nnz balloon | O(27 × L^2) |
| Bootstrap (dominant) | nbs × (histogram + L×nnz kernel + interp) | O(nbs × (n + L × nnz)) |
| **Total** | | **O(nbs × (n + L^2) + M^2 × L log L)** |

Here nnz (nonzero histogram bins) satisfies nnz ≤ L. The L×L pairwise
kernel matrix in CostFunction and the L×nnz bootstrap kernel are the most
expensive operations:

- **n < 1000**: L ≈ n → **O(nbs × n^2)** — quadratic in n
- **n > 1000**: L = 1000 → **O(nbs × n)** — linear in n (L^2 becomes constant)

### Summary table

| Function | n < 1000 (L ≈ n) | n > 1000 (L = 1000) | Dominant operation |
|---|---|---|---|
| `sshist` | O(n^2 log n) | O(n log n) | searchsorted loop |
| `sskernel` | O(nbs × n log n) | O(nbs × n) | bootstrap histograms |
| `ssvkernel` | O(nbs × n^2) | O(nbs × n) | CostFunction L×L matrix |

The L = 1000 cap is the key scaling boundary. Below it, `ssvkernel` scales
quadratically; above it, all functions scale linearly in n. For applications
with very large n, the bottleneck shifts from kernel evaluation to bootstrap
resampling (generating histograms from resampled data).
