# Harmonics: phase-modulated carrier decoding

Monte Carlo sweep that asks: given known carrier frequencies in 20–40 Hz,
white Gaussian noise (SNR = 0.20 power, −7 dB), and a finite sampling
window, how many simultaneous phase-coded carriers can be decoded at a
target accuracy?

## Method

Signal model: sum of `cos(2π f_i t + φ_i)` over `K` carriers, plus white
noise. Phases are drawn from `M` evenly-spaced bins (log2(M) bits per
carrier).

Decoder: quadrature matched filter. For each known frequency `f_i`,
project `y(t)` onto `cos` and `sin` bases, then `φ̂ = atan2(−Q, I)` and
snap to the nearest of `M` bins.

The sweep varies:
- window length: 50 ms – 2000 ms
- phase precision: 2, 4, 8, 16, 32, 64 levels
- carrier count: 1–80
- target accuracy: 0.90, 0.95, 0.98, 0.996

Trial count is scaled per target as `10 / (1 − target)` so each target
gets ~10 expected boundary misses (cheap targets run fewer trials).

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib
.venv/bin/python modulation-decode-acc.py
```

Outputs land in `phase_decode_results/`: raw CSV, per-target max-K
tables, accuracy-vs-K curves, and max-K heatmaps.

## Credits

- **GPT-5.5** — simulation design and implementation
- **Claude Opus 4.7** — refactor (per-target trial scaling) and this README
