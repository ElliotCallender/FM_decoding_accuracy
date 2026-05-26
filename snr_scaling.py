#!/usr/bin/env python3
"""
SNR scaling law sweep.

Holds window, phase precision, and target accuracy fixed.
Varies SNR over a log range and finds max simultaneous carriers K.
Fits a power law: K_max = a * SNR^alpha.
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader
_mod = SourceFileLoader("mdec", os.path.join(os.path.dirname(__file__), "modulation-decode-acc.py")).load_module()
simulate_condition = _mod.simulate_condition


FREQ_MIN = 20.0
FREQ_MAX = 40.0
WINDOW_SEC = 0.500
PHASE_LEVELS = 4
TARGET = 0.95
K_VALUES = list(range(1, 41))
SNR_POWERS = [3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]
SNR_MODE = "per_carrier_power"
FS = 1000.0
TRIAL_CHUNK = 500
RANDOM_SEED = 1
OUTDIR = "snr_scaling_results"


def trials_for(target: float) -> int:
    return int(round(10.0 / (1.0 - target)))


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    n_trials = trials_for(TARGET)

    rows = []
    for snr in SNR_POWERS:
        for k in K_VALUES:
            row = simulate_condition(
                k=k,
                window_sec=WINDOW_SEC,
                phase_levels=PHASE_LEVELS,
                snr_power=snr,
                snr_mode=SNR_MODE,
                fs=FS,
                n_trials=n_trials,
                trial_chunk=TRIAL_CHUNK,
                rng=rng,
            )
            row["snr_power"] = snr
            rows.append(row)
            print(f"SNR={snr:5.2f}  K={k:2d}  per_carrier={row['per_carrier_accuracy']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "snr_raw.csv"), index=False)

    # Max-K per SNR (prefix rule).
    summary = []
    for snr in SNR_POWERS:
        sub = df[df["snr_power"] == snr].sort_values("k_carriers")
        valid_k: int = 0
        for _, r in sub.iterrows():
            if r["per_carrier_accuracy"] >= TARGET:
                valid_k = int(r["k_carriers"])
            else:
                break
        summary.append({"snr_power": snr, "max_k": valid_k})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTDIR, "snr_max_k.csv"), index=False)
    print("\n=== max K vs SNR ===")
    print(summary_df.to_string(index=False))

    # Power-law fit on points with K >= 1.
    fit_df = summary_df[summary_df["max_k"] >= 1].copy()
    log_snr: np.ndarray = np.log(fit_df["snr_power"].to_numpy())
    log_k: np.ndarray = np.log(fit_df["max_k"].to_numpy())
    alpha, log_a = np.polyfit(log_snr, log_k, 1)
    a: float = float(np.exp(log_a))
    print(f"\nFit: K_max ≈ {a:.3f} * SNR^{alpha:.3f}")

    # Plot.
    snr_grid = np.logspace(np.log10(SNR_POWERS[0]), np.log10(SNR_POWERS[-1]), 100)
    k_fit = a * snr_grid ** alpha

    plt.figure(figsize=(7, 5))
    plt.loglog(summary_df["snr_power"], summary_df["max_k"], "o", markersize=8, label="measured")
    plt.loglog(snr_grid, k_fit, "--", label=f"fit: K = {a:.2f} · SNR^{alpha:.2f}")
    plt.xlabel("SNR (signal power / noise power)")
    plt.ylabel(f"Max K at ≥ {TARGET:.2f} per-carrier accuracy")
    plt.title(
        f"SNR scaling: T={WINDOW_SEC*1000:.0f} ms, "
        f"M={PHASE_LEVELS} ({math.log2(PHASE_LEVELS):.0f} bits), "
        f"{SNR_MODE}"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTDIR, "snr_scaling.png")
    plt.savefig(out_path, dpi=180)
    print(f"\nWrote plot: {out_path}")


if __name__ == "__main__":
    main()
