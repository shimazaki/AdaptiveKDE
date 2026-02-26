"""
AdaptiveKDE example — Old Faithful eruption durations.

Demonstrates all three density estimation methods on the classic
Old Faithful geyser dataset (107 eruption durations in minutes).

Usage:
    python example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from adaptivekde.sshist import sshist
from adaptivekde.sskernel import sskernel
from adaptivekde.ssvkernel import ssvkernel

# --- Data: Old Faithful eruption durations (minutes), 107 observations ---
# Source: Azzalini & Bowman (1990)
x = np.array([
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

# --- sshist: optimal histogram ---
# Finds the optimal number of bins by minimizing a cost function
# based on mean integrated squared error.
optN, optD, edges, _, _ = sshist(x)

# --- sskernel: fixed-bandwidth kernel density estimation ---
# Selects a single global bandwidth that minimizes MISE.
y_ss, t_ss, optw_ss, _, _, _, _ = sskernel(x)

# --- ssvkernel: locally adaptive kernel density estimation ---
# Selects a bandwidth that varies across the domain,
# adapting to local data density.
y_ssv, t_ssv, optw_ssv, _, _, _, _ = ssvkernel(x)

# --- Plot everything on one panel ---
fig, ax = plt.subplots(figsize=(8, 4.5))

# Histogram (normalized to density)
ax.hist(x, bins=edges, density=True, color="0.75", edgecolor="white",
        alpha=0.6, label=f"sshist ({optN} bins)")

# KDE curves
ax.plot(t_ss, y_ss, linewidth=1.5, label=f"sskernel (w={optw_ss:.3f})")
ax.plot(t_ssv, y_ssv, linewidth=1.5, label="ssvkernel (adaptive)")

# Rug plot — short ticks at the bottom showing each data point
ax.plot(x, np.zeros_like(x), "|", color="k", markersize=6, alpha=0.5)

ax.set_xlabel("Eruption duration (min)")
ax.set_ylabel("Density")
ax.set_title("Old Faithful eruption durations — AdaptiveKDE")
ax.legend()
fig.tight_layout()
plt.show()
