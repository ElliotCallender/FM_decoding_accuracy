#!/usr/bin/env python3
"""
Phase-modulated carrier decoding simulation.

Question modeled:
Given a known underlying frequency list between 20-40 Hz, Gaussian white noise,
and finite sampling windows, how many simultaneous phase-coded carrier waves can
be decoded at target accuracies?

Core assumptions:
- Each carrier encodes one discrete phase symbol.
- Frequencies are known to the decoder.
- Frequencies are evenly spaced between FREQ_MIN and FREQ_MAX for each carrier count K.
- Signal is a sum of cos(2πft + phase).
- Decoder uses matched sine/cosine projections at the known frequencies.
- "Accuracy" is measured in two ways:
    1. per_carrier_accuracy: fraction of individual carrier phase symbols decoded correctly.
    2. codeword_accuracy: fraction of trials where ALL K carrier phases are decoded correctly.

Important ambiguity:
"20% signal-to-noise ratio" can mean different things.
This script supports two useful conventions:

SNR_MODE = "per_carrier_power"
    Each carrier individually has signal_power / noise_power = 0.20.
    This isolates finite-window frequency leakage and phase precision effects.

SNR_MODE = "composite_power"
    The whole summed signal has signal_power / noise_power = 0.20.
    As K grows, total signal power grows and noise is scaled with it.
    This makes total-SNR fixed but per-carrier SNR falls with K.

Default is "per_carrier_power", which is usually the more relevant test if each
carrier is a separate channel competing against a fixed white-noise background.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# User-adjustable parameters
# -----------------------------

FREQ_MIN = 20.0
FREQ_MAX = 40.0

WINDOWS_SEC = [0.050, 0.100, 0.200, 0.500, 1.000, 2.000]
TARGET_ACCURACIES = [0.90, 0.95, 0.98, 0.996]

# Phase precision. M phase bins means log2(M) bits per carrier if M is a power of two.
PHASE_LEVELS = [2, 4, 8, 16, 32, 64]

# Interpreted as signal_power / noise_power.
SNR_POWER = 0.20

# Choose one:
#   "per_carrier_power"
#   "composite_power"
SNR_MODE = "per_carrier_power"

# Carrier count sweep.
# You can raise this if long windows and low phase precision allow many carriers.
K_VALUES = list(range(1, 81))

# Monte Carlo trial count per condition is scaled per target accuracy:
# n_trials(target) = 10 / (1 - target). Each target's sweep is run separately.
def trials_for(target: float) -> int:
    return int(round(10.0 / (1.0 - target)))

# Sampling rate. 1000 Hz is plenty for 20-40 Hz carriers.
FS = 1000.0

# Chunk size reduces memory load.
TRIAL_CHUNK = 500

RANDOM_SEED = 1
OUTDIR = "phase_decode_results"


# -----------------------------
# Simulation functions
# -----------------------------

def make_frequency_list(k: int, f_min: float, f_max: float) -> np.ndarray:
    """
    Evenly space k known carrier frequencies between f_min and f_max inclusive.

    For k=1, use the center frequency.
    """
    if k == 1:
        return np.array([(f_min + f_max) / 2.0], dtype=float)
    return np.linspace(f_min, f_max, k, dtype=float)


def nearest_phase_symbol(phi_hat: np.ndarray, m: int) -> np.ndarray:
    """
    Convert estimated phase in radians to nearest discrete phase symbol in [0, m-1].
    """
    two_pi = 2.0 * np.pi
    phi_wrapped = np.mod(phi_hat, two_pi)
    symbol_float = phi_wrapped / two_pi * m
    return np.mod(np.rint(symbol_float).astype(int), m)


def simulate_condition(
    *,
    k: int,
    window_sec: float,
    phase_levels: int,
    snr_power: float,
    snr_mode: str,
    fs: float,
    n_trials: int,
    trial_chunk: int,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate one condition and return per-carrier and full-codeword accuracy.

    Matched filter details:
    For each known frequency f, estimate phase by projecting y(t) onto cos(2πft)
    and sin(2πft). Since

        cos(ωt + φ) = cos(ωt)cos(φ) - sin(ωt)sin(φ),

    phase estimate is atan2(-<y,sin>, <y,cos>).
    """
    n_samples = int(round(window_sec * fs))
    t = np.arange(n_samples, dtype=float) / fs

    freqs = make_frequency_list(k, FREQ_MIN, FREQ_MAX)

    # Bases: shape [k, n_samples]
    angles = 2.0 * np.pi * freqs[:, None] * t[None, :]
    cos_basis = np.cos(angles)
    sin_basis = np.sin(angles)

    # One carrier with amplitude 1 has average power ~1/2.
    per_carrier_signal_power = 0.5

    if snr_mode == "per_carrier_power":
        noise_variance = per_carrier_signal_power / snr_power
    elif snr_mode == "composite_power":
        composite_signal_power = k * per_carrier_signal_power
        noise_variance = composite_signal_power / snr_power
    else:
        raise ValueError("snr_mode must be 'per_carrier_power' or 'composite_power'")

    noise_sigma = math.sqrt(noise_variance)

    correct_carriers = 0
    total_carriers = 0
    correct_codewords = 0
    total_codewords = 0

    # Useful diagnostic: spacing in Hz.
    if k == 1:
        freq_spacing = np.nan
    else:
        freq_spacing = (FREQ_MAX - FREQ_MIN) / (k - 1)

    for start in range(0, n_trials, trial_chunk):
        chunk = min(trial_chunk, n_trials - start)

        # Random discrete phase symbols, then phase radians.
        symbols = rng.integers(0, phase_levels, size=(chunk, k))
        phases = 2.0 * np.pi * symbols / phase_levels

        # Build signal:
        # cos(wt + phi) = cos(wt)cos(phi) - sin(wt)sin(phi)
        signal = (
            np.cos(phases) @ cos_basis
            - np.sin(phases) @ sin_basis
        )

        noise = rng.normal(0.0, noise_sigma, size=signal.shape)
        y = signal + noise

        # Matched projections: shapes [chunk, k]
        i_proj = y @ cos_basis.T
        q_proj = y @ sin_basis.T

        phase_hat = np.arctan2(-q_proj, i_proj)
        decoded = nearest_phase_symbol(phase_hat, phase_levels)

        correct_matrix = decoded == symbols
        correct_carriers += int(correct_matrix.sum())
        total_carriers += int(correct_matrix.size)

        correct_codewords += int(np.all(correct_matrix, axis=1).sum())
        total_codewords += chunk

    per_carrier_accuracy = correct_carriers / total_carriers
    codeword_accuracy = correct_codewords / total_codewords

    return {
        "k_carriers": k,
        "window_ms": int(round(window_sec * 1000)),
        "window_sec": window_sec,
        "phase_levels": phase_levels,
        "phase_bits": math.log2(phase_levels),
        "snr_power": snr_power,
        "snr_db": 10.0 * math.log10(snr_power),
        "snr_mode": snr_mode,
        "freq_min_hz": FREQ_MIN,
        "freq_max_hz": FREQ_MAX,
        "freq_spacing_hz": freq_spacing,
        "cycles_at_20hz": 20.0 * window_sec,
        "cycles_at_40hz": 40.0 * window_sec,
        "fourier_resolution_hz": 1.0 / window_sec,
        "n_trials": n_trials,
        "fs_hz": fs,
        "per_carrier_accuracy": per_carrier_accuracy,
        "codeword_accuracy": codeword_accuracy,
    }


def run_sweep(n_trials: int, target: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    total = len(WINDOWS_SEC) * len(PHASE_LEVELS) * len(K_VALUES)
    done = 0

    for window_sec in WINDOWS_SEC:
        for phase_levels in PHASE_LEVELS:
            for k in K_VALUES:
                done += 1
                print(
                    f"[target={target:.3f} n={n_trials}] "
                    f"[{done:4d}/{total}] "
                    f"T={window_sec*1000:5.0f} ms, "
                    f"M={phase_levels:2d}, "
                    f"K={k:2d}"
                )
                row = simulate_condition(
                    k=k,
                    window_sec=window_sec,
                    phase_levels=phase_levels,
                    snr_power=SNR_POWER,
                    snr_mode=SNR_MODE,
                    fs=FS,
                    n_trials=n_trials,
                    trial_chunk=TRIAL_CHUNK,
                    rng=rng,
                )
                row["target_accuracy"] = target
                rows.append(row)

    return pd.DataFrame(rows)


def run_all_sweeps() -> pd.DataFrame:
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    frames = []
    for target in TARGET_ACCURACIES:
        n_trials = trials_for(target)
        frames.append(run_sweep(n_trials=n_trials, target=target, rng=rng))
    df = pd.concat(frames, ignore_index=True)
    raw_path = os.path.join(OUTDIR, "raw_accuracy_by_condition.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nWrote raw results: {raw_path}")
    return df


def max_k_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    For each window, phase precision, and target accuracy, find the largest K
    whose accuracy is at least the target.

    Because Monte Carlo estimates are noisy and accuracy is not always perfectly
    monotonic in K, this uses a conservative prefix rule:
    max K such that all tested K' <= K meet the target.

    This avoids reporting a high-K lucky blip after a lower-K failure.
    """
    records = []

    for phase_levels in sorted(df["phase_levels"].unique()):
        for window_ms in sorted(df["window_ms"].unique()):
            for target in TARGET_ACCURACIES:
                sub = df[
                    (df["phase_levels"] == phase_levels)
                    & (df["window_ms"] == window_ms)
                    & (df["target_accuracy"] == target)
                ].sort_values("k_carriers")

                valid_k = 0
                for _, row in sub.iterrows():
                    if row[metric] >= target:
                        valid_k = int(row["k_carriers"])
                    else:
                        break

                records.append({
                    "metric": metric,
                    "phase_levels": phase_levels,
                    "phase_bits": math.log2(phase_levels),
                    "window_ms": window_ms,
                    "target_accuracy": target,
                    "max_simultaneous_carriers": valid_k,
                })

    return pd.DataFrame(records)


def write_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_carrier = max_k_table(df, "per_carrier_accuracy")
    codeword = max_k_table(df, "codeword_accuracy")

    per_path = os.path.join(OUTDIR, "answer_table_per_carrier_accuracy.csv")
    code_path = os.path.join(OUTDIR, "answer_table_whole_codeword_accuracy.csv")

    per_carrier.to_csv(per_path, index=False)
    codeword.to_csv(code_path, index=False)

    print(f"Wrote answer table: {per_path}")
    print(f"Wrote answer table: {code_path}")

    return per_carrier, codeword


# -----------------------------
# Plotting
# -----------------------------

def plot_accuracy_curves(df: pd.DataFrame, metric: str) -> None:
    """
    Line plots: accuracy vs K, one plot per phase precision and window.
    Uses the strictest-target sweep (highest n_trials) for the lowest-noise curve.
    """
    df = df[df["target_accuracy"] == max(TARGET_ACCURACIES)]
    for phase_levels in sorted(df["phase_levels"].unique()):
        plt.figure(figsize=(10, 6))

        for window_ms in sorted(df["window_ms"].unique()):
            sub = df[
                (df["phase_levels"] == phase_levels)
                & (df["window_ms"] == window_ms)
            ].sort_values("k_carriers")

            plt.plot(
                sub["k_carriers"],
                sub[metric],
                marker="o",
                markersize=3,
                linewidth=1,
                label=f"{window_ms} ms",
            )

        for target in TARGET_ACCURACIES:
            plt.axhline(target, linestyle="--", linewidth=0.8)

        plt.ylim(0.0, 1.01)
        plt.xlabel("Number of simultaneous carriers, K")
        plt.ylabel(metric.replace("_", " "))
        plt.title(
            f"{metric.replace('_', ' ').title()} vs carrier count\n"
            f"{phase_levels} phase levels "
            f"({math.log2(phase_levels):.0f} bits/carrier), "
            f"SNR={SNR_POWER} power ratio ({10*math.log10(SNR_POWER):.1f} dB), "
            f"{SNR_MODE}"
        )
        plt.legend(title="Window")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(
            OUTDIR,
            f"{metric}_curves_M{phase_levels}.png"
        )
        plt.savefig(path, dpi=180)
        plt.close()
        print(f"Wrote plot: {path}")


def plot_max_k_heatmaps(answer_df: pd.DataFrame, metric: str) -> None:
    """
    Heatmaps: max K by window and phase precision, one plot per target accuracy.
    """
    for target in TARGET_ACCURACIES:
        sub = answer_df[answer_df["target_accuracy"] == target]

        pivot = sub.pivot(
            index="phase_levels",
            columns="window_ms",
            values="max_simultaneous_carriers",
        ).sort_index()

        plt.figure(figsize=(9, 5))
        image = plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.colorbar(image, label="Max simultaneous carriers")

        plt.xticks(
            ticks=np.arange(len(pivot.columns)),
            labels=[str(c) for c in pivot.columns],
        )
        plt.yticks(
            ticks=np.arange(len(pivot.index)),
            labels=[str(i) for i in pivot.index],
        )

        # Annotate cells.
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                val = pivot.values[y, x]
                plt.text(x, y, str(int(val)), ha="center", va="center")

        plt.xlabel("Sampling window, ms")
        plt.ylabel("Phase levels")
        plt.title(
            f"Max K at ≥ {100*target:.1f}% {metric.replace('_', ' ')}\n"
            f"SNR={SNR_POWER} power ratio ({10*math.log10(SNR_POWER):.1f} dB), "
            f"{SNR_MODE}"
        )
        plt.tight_layout()

        path = os.path.join(
            OUTDIR,
            f"max_k_heatmap_{metric}_target_{str(target).replace('.', 'p')}.png"
        )
        plt.savefig(path, dpi=180)
        plt.close()
        print(f"Wrote plot: {path}")


def plot_frequency_spacing_diagnostics(df: pd.DataFrame) -> None:
    """
    Plot max K roughly predicted by Fourier resolution spacing.

    This is not a decoding result; it helps interpret brittleness.
    A sampling window T has natural spectral resolution ~1/T Hz.
    In a 20 Hz band, the number of cleanly separated bins is roughly 20T.
    """
    windows = np.array(WINDOWS_SEC)
    naive_bins = (FREQ_MAX - FREQ_MIN) * windows

    plt.figure(figsize=(8, 5))
    plt.plot(windows * 1000, naive_bins, marker="o")
    plt.xlabel("Sampling window, ms")
    plt.ylabel("Bandwidth × window = rough independent Fourier bins")
    plt.title("Rough frequency degrees of freedom in 20–40 Hz band")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTDIR, "frequency_resolution_diagnostic.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Wrote plot: {path}")


def main() -> None:
    print("Running phase decoding simulation.")
    print(f"SNR_POWER = {SNR_POWER} ({10*math.log10(SNR_POWER):.2f} dB)")
    print(f"SNR_MODE  = {SNR_MODE}")
    print(f"Trials per target: " + ", ".join(
        f"{t:.3f}->{trials_for(t)}" for t in TARGET_ACCURACIES
    ))
    print(f"Output    = {OUTDIR}/\n")

    df = run_all_sweeps()
    per_carrier_table, codeword_table = write_tables(df)

    plot_accuracy_curves(df, "per_carrier_accuracy")
    plot_accuracy_curves(df, "codeword_accuracy")

    plot_max_k_heatmaps(per_carrier_table, "per_carrier_accuracy")
    plot_max_k_heatmaps(codeword_table, "codeword_accuracy")

    plot_frequency_spacing_diagnostics(df)

    # Print compact answer tables to terminal.
    print("\n=== Max simultaneous carriers: per-carrier accuracy ===")
    for phase_levels in PHASE_LEVELS:
        print(f"\nPhase levels = {phase_levels} ({math.log2(phase_levels):.0f} bits/carrier)")
        sub = per_carrier_table[per_carrier_table["phase_levels"] == phase_levels]
        pretty = sub.pivot(
            index="window_ms",
            columns="target_accuracy",
            values="max_simultaneous_carriers",
        )
        print(pretty.to_string())

    print("\n=== Max simultaneous carriers: whole-codeword accuracy ===")
    for phase_levels in PHASE_LEVELS:
        print(f"\nPhase levels = {phase_levels} ({math.log2(phase_levels):.0f} bits/carrier)")
        sub = codeword_table[codeword_table["phase_levels"] == phase_levels]
        pretty = sub.pivot(
            index="window_ms",
            columns="target_accuracy",
            values="max_simultaneous_carriers",
        )
        print(pretty.to_string())

    print("\nDone.")
    print(f"Inspect CSVs and PNGs in: {OUTDIR}/")


if __name__ == "__main__":
    main()
