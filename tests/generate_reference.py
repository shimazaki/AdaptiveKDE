"""Generate golden reference data for AdaptiveKDE tests.

Run once with: conda run -n basic39 python tests/generate_reference.py

Uses the Old Faithful eruption-duration dataset (107 bimodal points).
"""
import os
import numpy as np
from adaptivekde.sshist import sshist
from adaptivekde.sskernel import sskernel
from adaptivekde.ssvkernel import ssvkernel

# Old Faithful eruption durations (minutes) — 107 observations
# Source: Azzalini & Bowman (1990), widely used bimodal benchmark
OLD_FAITHFUL = np.array([
    1.80, 1.80, 1.80, 1.85, 1.87, 1.90, 1.92, 1.93, 1.93, 1.95,
    1.95, 1.97, 2.00, 2.03, 2.05, 2.07, 2.08, 2.10, 2.10, 2.13,
    2.15, 2.17, 2.18, 2.20, 2.22, 2.23, 2.25, 2.27, 2.30, 2.30,
    2.33, 2.33, 2.35, 2.37, 2.40, 2.42, 2.43, 2.45, 2.47, 2.50,
    2.50, 2.52, 2.55, 2.57, 2.58, 2.60, 2.63, 2.65, 2.67, 2.70,
    2.72, 2.73, 2.75, 2.78, 2.80, 2.83, 2.85, 2.87, 2.90, 2.92,
    2.93, 2.95, 2.97, 3.00, 3.02, 3.03, 3.05, 3.07, 3.08, 3.10,
    3.15, 3.17, 3.20, 3.22, 3.25, 3.27, 3.30, 3.33, 3.35, 3.37,
    3.40, 3.42, 3.45, 3.47, 3.50, 3.53, 3.55, 3.57, 3.60, 3.63,
    3.65, 3.67, 3.70, 3.72, 3.75, 3.78, 3.80, 3.83, 3.85, 3.88,
    3.90, 3.93, 3.95, 3.98, 4.00, 4.02, 4.05,
])

OUT_DIR = os.path.join(os.path.dirname(__file__), "reference_data")
os.makedirs(OUT_DIR, exist_ok=True)

x = OLD_FAITHFUL.copy()

# --- sshist (deterministic) ---
print("Generating sshist reference...")
optN, optD, edges, C, N = sshist(x)
np.savez(
    os.path.join(OUT_DIR, "sshist_ref.npz"),
    optN=optN,
    optD=optD,
    edges=edges,
    C=C,
)
print(f"  optN={optN}, optD={optD:.6f}, len(edges)={len(edges)}, len(C)={len(C)}")

# --- sskernel (seeded bootstrap, nbs=200) ---
print("Generating sskernel reference...")
np.random.seed(0)
y, t, optw, W, C, confb95, yb = sskernel(x, nbs=200)
np.savez(
    os.path.join(OUT_DIR, "sskernel_ref.npz"),
    y=y,
    t=t,
    optw=optw,
    W=W,
    C=C,
    confb95=confb95,
    yb=yb,
)
print(f"  optw={optw:.6f}, len(y)={len(y)}, yb.shape={yb.shape}")

# --- ssvkernel (seeded bootstrap, nbs=50) ---
print("Generating ssvkernel reference...")
np.random.seed(0)
y, t, optw, gs, C, confb95, yb = ssvkernel(x, nbs=50)
np.savez(
    os.path.join(OUT_DIR, "ssvkernel_ref.npz"),
    y=y,
    t=t,
    optw=optw,
    gs=gs,
    C=C,
    confb95=confb95,
    yb=yb,
)
print(f"  len(y)={len(y)}, gs.shape={gs.shape}, yb.shape={yb.shape}")

print("\nAll reference data saved to", OUT_DIR)
