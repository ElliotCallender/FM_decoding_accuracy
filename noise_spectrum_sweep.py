#!/usr/bin/env python3
"""
Noise-power × window × spectrum sweep.

Fixes phase precision (4 levels = 2 bits/carrier) and target accuracy (0.996).
Varies window length over 100-500 ms, SNR (signal_power / noise_power) over a
log range, and noise spectrum ("white" vs "pink").

Pink-noise SNR is calibrated at F_REF = sqrt(20 * 40) Hz; carriers below F_REF
see worse SNR, carriers above see better.

Produces:
  noise_results/raw.csv
  noise_results/max_k.csv
  noise_results/max_k_vs_snr_white.png
  noise_results/max_k_vs_snr_pink.png
  noise_results/max_k_vs_snr_compare.png
  noise_results/per_carrier_heatmap_<spectrum>_T<ms>.png
"""

import math
import os
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_mdec = SourceFileLoader(
    "mdec",
    os.path.join(os.path.dirname(__file__), "modulation-decode-acc.py"),
).load_module()
simulate_condition = _mdec.simulate_condition
F_REF = _mdec.F_REF


WINDOWS_SEC = [0.100, 0.200, 0.300, 0.400, 0.500]
PHASE_LEVELS = 4
TARGET = 0.996
K_VALUES = list(range(1, 31))
SNR_POWERS = [3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]
SPECTRA = ["white", "pink"]
SNR_MODE = "per_carrier_power"
FS = 1000.0
TRIAL_CHUNK = 500
RANDOM_SEED = 1
OUTDIR = "noise_results"


def trials_for(target: float) -> int:
    return int(round(10.0 / (1.0 - target)))


def run() -> pd.DataFrame:
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    n_trials = trials_for(TARGET)
    rows = []
    total = len(SPECTRA) * len(WINDOWS_SEC) * len(SNR_POWERS) * len(K_VALUES)
    done = 0
    for spectrum in SPECTRA:
        for window_sec in WINDOWS_SEC:
            for snr in SNR_POWERS:
                for k in K_VALUES:
                    done += 1
                    row = simulate_condition(
                        k=k,
                        window_sec=window_sec,
                        phase_levels=PHASE_LEVELS,
                        snr_power=snr,
                        snr_mode=SNR_MODE,
                        fs=FS,
                        n_trials=n_trials,
                        trial_chunk=TRIAL_CHUNK,
                        rng=rng,
                        noise_spectrum=spectrum,
                        f_ref=F_REF,
                    )
                    rows.append(row)
                    if done % 50 == 0 or done == total:
                        print(
                            f"[{done:5d}/{total}] {spectrum:5s} "
                            f"T={window_sec*1000:5.0f}ms SNR={snr:7.4f} K={k:2d} "
                            f"acc={row['per_carrier_accuracy']:.3f}"
                        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "raw.csv"), index=False)
    return df


def max_k_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for spectrum in SPECTRA:
        for window_sec in WINDOWS_SEC:
            for snr in SNR_POWERS:
                sub = df[
                    (df["noise_spectrum"] == spectrum)
                    & (df["window_sec"] == window_sec)
                    & (df["snr_power"] == snr)
                ].sort_values("k_carriers")
                valid_k: int = 0
                for _, r in sub.iterrows():
                    if r["per_carrier_accuracy"] >= TARGET:
                        valid_k = int(r["k_carriers"])
                    else:
                        break
                records.append({
                    "noise_spectrum": spectrum,
                    "window_sec": window_sec,
                    "window_ms": int(round(window_sec * 1000)),
                    "snr_power": snr,
                    "max_k": valid_k,
                })
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUTDIR, "max_k.csv"), index=False)
    return out


def plot_max_k_vs_snr(max_df: pd.DataFrame, spectrum: str) -> None:
    sub = max_df[max_df["noise_spectrum"] == spectrum]
    plt.figure(figsize=(8, 5.5))
    for window_sec in WINDOWS_SEC:
        ss = sub[sub["window_sec"] == window_sec].sort_values("snr_power")
        plt.semilogx(
            ss["snr_power"], ss["max_k"],
            marker="o", linewidth=1.5,
            label=f"T = {int(window_sec*1000)} ms",
        )
    plt.xlabel("SNR (signal power / noise power)")
    plt.ylabel(f"Max K at ≥ {TARGET:.3f} per-carrier accuracy")
    plt.title(
        f"Max simultaneous carriers vs SNR — {spectrum} noise\n"
        f"M = {PHASE_LEVELS} ({math.log2(PHASE_LEVELS):.0f} bits/carrier), "
        f"f_ref = {F_REF:.2f} Hz"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTDIR, f"max_k_vs_snr_{spectrum}.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Wrote {path}")


def plot_max_k_compare(max_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(WINDOWS_SEC), figsize=(4 * len(WINDOWS_SEC), 4.5), sharey=True)
    for ax, window_sec in zip(axes, WINDOWS_SEC):
        for spectrum in SPECTRA:
            sub = max_df[
                (max_df["noise_spectrum"] == spectrum)
                & (max_df["window_sec"] == window_sec)
            ].sort_values("snr_power")
            ax.semilogx(
                sub["snr_power"], sub["max_k"],
                marker="o", linewidth=1.5,
                label=spectrum,
            )
        ax.set_title(f"T = {int(window_sec*1000)} ms")
        ax.set_xlabel("SNR (power)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(f"Max K at ≥ {TARGET:.3f}")
    fig.suptitle("White vs pink noise: max simultaneous carriers")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "max_k_vs_snr_compare.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_per_carrier_heatmaps(df: pd.DataFrame) -> None:
    """Per-carrier accuracy as a heatmap over (SNR, K) for each window and spectrum."""
    for spectrum in SPECTRA:
        for window_sec in WINDOWS_SEC:
            sub = df[
                (df["noise_spectrum"] == spectrum)
                & (df["window_sec"] == window_sec)
            ]
            pivot = sub.pivot(
                index="snr_power", columns="k_carriers", values="per_carrier_accuracy"
            ).sort_index()

            fig, ax = plt.subplots(figsize=(9, 5.5))
            im = ax.imshow(
                pivot.values,
                aspect="auto", origin="lower",
                vmin=0.0, vmax=1.0,
                extent=[
                    pivot.columns.min() - 0.5, pivot.columns.max() + 0.5,
                    0, len(pivot.index),
                ],
            )
            ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
            ax.set_yticklabels([f"{s:.3g}" for s in pivot.index])
            ax.set_xlabel("K (simultaneous carriers)")
            ax.set_ylabel("SNR (power)")
            ax.set_title(
                f"Per-carrier accuracy — {spectrum} noise, T = {int(window_sec*1000)} ms\n"
                f"M = {PHASE_LEVELS}, f_ref = {F_REF:.2f} Hz"
            )
            fig.colorbar(im, ax=ax, label="per-carrier accuracy")
            fig.tight_layout()
            path = os.path.join(
                OUTDIR,
                f"per_carrier_heatmap_{spectrum}_T{int(window_sec*1000)}.png",
            )
            fig.savefig(path, dpi=180)
            plt.close(fig)
            print(f"Wrote {path}")


def main() -> None:
    print(f"Sweep: {len(SPECTRA)} spectra x {len(WINDOWS_SEC)} windows x "
          f"{len(SNR_POWERS)} SNRs x {len(K_VALUES)} K values, "
          f"n_trials = {trials_for(TARGET)} each.")
    df = run()
    max_df = max_k_table(df)

    print("\n=== max K table ===")
    print(max_df.to_string(index=False))

    for spectrum in SPECTRA:
        plot_max_k_vs_snr(max_df, spectrum)
    plot_max_k_compare(max_df)
    plot_per_carrier_heatmaps(df)
    print(f"\nDone. Outputs in {OUTDIR}/")


if __name__ == "__main__":
    main()
