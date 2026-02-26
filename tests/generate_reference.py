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
    4.37, 3.87, 4.00, 4.03, 3.50, 4.08, 2.25, 4.70, 1.73, 4.93,
    1.73, 4.62, 3.43, 4.25, 1.68, 3.92, 3.68, 3.10, 4.03, 1.77,
    4.08, 1.75, 3.20, 1.85, 4.62, 1.97, 4.50, 3.92, 4.35, 2.33,
    3.83, 1.88, 4.60, 1.80, 4.73, 1.77, 4.57, 1.85, 3.52, 4.00,
    3.70, 3.72, 4.25, 3.58, 3.80, 3.77, 3.75, 2.50, 4.50, 4.10,
    3.70, 3.80, 3.43, 4.00, 2.27, 4.40, 4.05, 4.25, 3.33, 2.00,
    4.33, 2.93, 4.58, 1.90, 3.58, 3.73, 3.73, 1.82, 4.63, 3.50,
    4.00, 3.67, 1.67, 4.60, 1.67, 4.00, 1.80, 4.42, 1.90, 4.63,
    2.93, 3.50, 1.97, 4.28, 1.83, 4.13, 1.83, 4.65, 4.20, 3.93,
    4.33, 1.83, 4.53, 2.03, 4.18, 4.43, 4.07, 4.13, 3.95, 4.10,
    2.72, 4.58, 1.90, 4.50, 1.95, 4.83, 4.12,
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
