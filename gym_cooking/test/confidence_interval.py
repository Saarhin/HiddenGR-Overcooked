import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# Configuration ― change these to match your real setup
# ------------------------------------------------------------
DATA_DIR = Path("/Users/sarahamini/Desktop/Sarah/Github/Hidden_results/results/")          # Folder that contains your *.npy files
FILE_PATTERN = "policies_Summary_seed{}_fetcherDQN_simpleChef5x5/rewards.csv"  # e.g. seed0_returns.npy … seed9_returns.npy
NUM_SEEDS = 10
MOVING_AVG_WINDOW = 1000           # e.g. 1 000‑episode moving average
BOOTSTRAP_REPS = 1000              # Increase for tighter CI (⇧ runtime)
ALPHA = 0.05                       # For a (1‑α)=95 % CI
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ------------------------------------------------------------

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average using convolution with 'valid' mode."""
    return np.convolve(x, np.ones(w, dtype=float) / w, mode="valid")

returns_list = []

# Attempt to load real data
for i in SEEDS:
    f = DATA_DIR / FILE_PATTERN.format(i)
    if f.exists():
        col = pd.read_csv(f)['reward'].to_numpy()
        returns_list.append(col)
    else:
        # ------------------------------------------------------------
        # Demo fallback: create a synthetic learning curve so that the
        # example runs even when *.npy files are not present.
        #
        # Remove this entire block when you have real files!
        # ------------------------------------------------------------
        print(f"[demo] {f} not found – generating synthetic data for seed {i}")
        EPISODES = 5_000
        np.random.seed(i)
        trend = np.linspace(0, 1, EPISODES)           # slow improvement
        noise = np.random.randn(EPISODES) * 0.2
        returns_list.append(trend + noise)
        # ------------------------------------------------------------

# Sanity‑check all seeds have the same number of episodes
lengths = {len(r) for r in returns_list}
if len(lengths) != 1:
    raise ValueError(f"All seeds must have the same #episodes. Got lengths: {lengths}")
EPISODES = lengths.pop()

# ------------------------------------------------------------
# 1. Moving average per seed
# ------------------------------------------------------------
ma_per_seed = np.vstack([moving_average(r, MOVING_AVG_WINDOW) for r in returns_list])
n_points = ma_per_seed.shape[1]
episodes_axis = np.arange(MOVING_AVG_WINDOW - 1, EPISODES)

# ------------------------------------------------------------
# 2. Mean learning curve across seeds
# ------------------------------------------------------------
mean_curve = ma_per_seed.mean(axis=0)

# ------------------------------------------------------------
# 3. Bootstrap (non‑parametric) CI across seeds, point‑wise
# ------------------------------------------------------------
lower = np.empty(n_points)
upper = np.empty(n_points)

rng = np.random.default_rng(seed=42)
for t in range(n_points):
    sample = ma_per_seed[:, t]                      # shape: (NUM_SEEDS,)
    # Draw BOOTSTRAP_REPS resamples (with replacement) of length NUM_SEEDS
    resamples = rng.choice(sample, size=(BOOTSTRAP_REPS, NUM_SEEDS), replace=True)
    boot_means = resamples.mean(axis=1)
    lower[t] = np.percentile(boot_means, 100 * (ALPHA / 2))
    upper[t] = np.percentile(boot_means, 100 * (1 - ALPHA / 2))

# ------------------------------------------------------------
# 4. Plot
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(episodes_axis, mean_curve, label="Mean return (moving avg)")
plt.fill_between(episodes_axis, lower, upper, alpha=0.3,
                 label=f"{int((1-ALPHA)*100)}% bootstrap CI")
plt.xlabel("Episode")
plt.ylabel("Return (moving average)")
plt.title("Multi Agent - 5x5 map - Water")
plt.legend()
plt.tight_layout()
plt.show()