# Harmonics: phase-modulated carrier decoding

Monte Carlo sweeps that ask: given known carrier frequencies in 20–40 Hz,
additive noise, and a finite sampling window, how many simultaneous
phase-coded carriers can be decoded at a target accuracy?

## Method

Signal model: sum of `cos(2π f_i t + φ_i)` over `K` carriers, plus additive
Gaussian noise. Phases are drawn from `M` evenly-spaced bins (log2(M) bits
per carrier).

Decoder: quadrature matched filter. For each known frequency `f_i`,
project `y(t)` onto `cos` and `sin` bases, then `φ̂ = atan2(−Q, I)` and
snap to the nearest of `M` bins.

SNR convention: `SNR = signal_power / noise_power` (linear power ratio,
per carrier). Noise spectrum is selectable:

- **white** (default): flat PSD, plain `N(0, σ)` samples.
- **pink**: `1/f` power spectrum (`1/√f` amplitude). Calibrated so that
  the stated SNR holds exactly at `F_REF = √(20·40) ≈ 28.28 Hz`. Carriers
  below `F_REF` see proportionally more noise (variance scales as
  `F_REF/f`); carriers above see less.

Trial count is scaled per target as `10 / (1 − target)` so each target
gets ~10 expected boundary misses.

## Scripts

### `modulation-decode-acc.py` — main timing sweep

Varies window (50 ms – 2 s), phase precision (2–64 levels), carrier
count (1–80), and target accuracy (0.90 – 0.996) at fixed SNR = 0.20.
Default noise is white; flip `NOISE_SPECTRUM = "pink"` at the top of the
file to rerun under pink noise.

Outputs → `timing_results/`: `raw_accuracy_by_condition.csv`,
`answer_table_*.csv`, accuracy-vs-K curves and max-K heatmaps.

### `snr_scaling.py` — SNR scaling law (one window, one M)

Sweeps SNR over a log range at fixed window and phase precision to fit
`K_max ≈ a · SNR^α`. Reveals two regimes: noise-limited (steep) and
leakage-limited (flat).

Outputs → `snr_scaling_results/`: `snr_raw.csv`, `snr_max_k.csv`,
`snr_scaling.png`.

### `noise_spectrum_sweep.py` — noise power × window × spectrum

At fixed phase precision (2 bits) and target accuracy (0.996), sweeps
window length (100 – 500 ms), SNR (3·10⁻³ – 2.0), and noise spectrum
(`white`, `pink`) independently. Useful for comparing how the
biologically realistic 1/f noise floor shifts max-K relative to white.

Outputs → `noise_results/`: `raw.csv`, `max_k.csv`,
`max_k_vs_snr_<spectrum>.png`, `max_k_vs_snr_compare.png`,
`per_carrier_heatmap_<spectrum>_T<ms>.png`.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib
.venv/bin/python modulation-decode-acc.py      # main timing sweep
.venv/bin/python snr_scaling.py                # SNR scaling
.venv/bin/python noise_spectrum_sweep.py       # noise spectrum sweep
```

The default run of `modulation-decode-acc.py` (no flags, no edits) is
the original white-noise sweep — the new spectrum switch is opt-in.

## Credits

- **GPT-5.5** — original simulation design and implementation
- **Claude Opus 4.7** — per-target trial scaling, SNR scaling sweep,
  pink-noise spectrum, noise×window sweep, and this README
