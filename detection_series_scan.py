"""
detection_series_scan.py

Detection-ONLY version of peak_detection_loop.ipynb, looped over every
temperature column in a wide-format CSV. No `refine_grouped` / non-linear
fitting -- this is purely for visually scanning how the FCN's raw detections
(after raw-signal filtering) evolve across the temperature series, before
committing to any fitting strategy.

Reuses the exact detection + filtering logic from peak_detection_loop.ipynb
sections 1-3:
  1. resample_to_training_grid
  2. savgol_filter smoothing (detection only, not for downstream fitting)
  3. predict_peaks_dense (FCN inference)
  4. compute_multiscale_diff_movavg + threshold gating (raw-signal filter)

Produces:
  - detected_peaks_summary.csv   : T, n_peaks, peak positions (per T)
  - detection_grid.pdf           : small-multiples grid, one panel per T,
                                    raw spectrum + detected peak markers
  - detection_waterfall.pdf      : stacked/offset waterfall of all spectra
                                    with detected peak markers, for tracking
                                    shift/broadening/merging by eye

Usage:
    python detection_series_scan.py \
        --csv data/.dev/T-depend\ Raman\ of\ Agv-VV/T-depend\ Raman-Agv-VV.csv \
        --model saved_models/dense_model.pt \
        --config saved_models/dense_model_config.json \
        --out-dir detection_out/

"""

import os
import re
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from src.inference import load_model, predict_peaks_dense, resample_to_training_grid
from processing.peak_filters_by_raw_signal import (compute_multiscale_diff_movavg,
                                                   filter_by_raw_signal_flat)


# ----------------------------------------------------------------------
# 1. Load wide-format CSV
# ----------------------------------------------------------------------

def load_temperature_series(csv_path):
    """Returns (freq, temps_K [ascending], intensity[n_freq, n_temps], col_names)."""
    df = pd.read_csv(csv_path)
    freq_col = df.columns[0]
    temp_cols = list(df.columns[1:])

    def parse_T(col):
        m = re.search(r"[-+]?\d*\.?\d+", col)
        if not m:
            raise ValueError(f"Could not parse a temperature from column name: {col!r}")
        return float(m.group())

    temps = np.array([parse_T(c) for c in temp_cols])
    order = np.argsort(temps)
    temps_sorted = temps[order]
    cols_sorted = [temp_cols[i] for i in order]

    freq = df[freq_col].to_numpy(dtype=float)
    intensity = df[cols_sorted].to_numpy(dtype=float)

    return freq, temps_sorted, intensity, cols_sorted


# ----------------------------------------------------------------------
# 2. Detection only (mirrors peak_detection_loop.ipynb sections 1-3)
# ----------------------------------------------------------------------

def detect_peaks_single(
    real_freq, real_intensity, cnn_model, cfg, device,
    detect_threshold=0.6,
    window_bank=(11, 35, 71),
    signal_threshold=0.6,
    average_window=6
):
    """Returns (raw_resampled, detections_filtered) where detections_filtered
    is a list of (conf, A, pos, gamma) -- FCN detections that passed the
    raw-signal check. No non-linear fitting is performed."""
    raw_resampled = resample_to_training_grid(real_freq, real_intensity, cfg.W)
    smooth_for_detect = savgol_filter(raw_resampled, window_length=average_window, polyorder=1)

    detected = predict_peaks_dense(cnn_model, smooth_for_detect, threshold=detect_threshold,
                                    cfg=cfg, device=device)

    scale_profiles = compute_multiscale_diff_movavg(
        raw_resampled, detected, list(window_bank), cfg.W, span=30, scale=200
    )

    detections_filtered = []
    for (conf, A, pos, gamma), profile in zip(detected, scale_profiles):
        if np.any(profile > signal_threshold):
            detections_filtered.append((conf, A, pos, gamma))

    fixed_threshold = 0.2
    detections_filtered = filter_by_raw_signal_flat(detections_filtered, cfg.W, raw_resampled, cfg, threshold=fixed_threshold)

    return raw_resampled, detections_filtered


# ----------------------------------------------------------------------
# 3. Driver: loop over all temperatures, detection only
# ----------------------------------------------------------------------

def run_detection_scan(csv_path, model_path, config_path, device=None, verbose=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model, cfg, device = load_model(model_path, config_path, device=device, mode="eval")

    real_freq, temps, intensity, col_names = load_temperature_series(csv_path)

    spectra_resampled = []
    detections_per_T = []
    summary_rows = []

    window_bank = (11, 35, 71)
    signal_threshold = 0.2

    for k, (T, col) in enumerate(zip(temps, col_names)):
        if verbose:
            print(f"[{k+1}/{len(temps)}] T = {T} K ({col})")

        real_intensity = intensity[:, k]
        raw_resampled, detections_filtered = detect_peaks_single(
            real_freq, real_intensity, cnn_model, cfg, device,
            window_bank=window_bank,
            signal_threshold=signal_threshold,
            average_window=6
        )

        spectra_resampled.append(raw_resampled)
        detections_per_T.append(detections_filtered)

        positions = sorted(p[2] for p in detections_filtered)  # (conf, A, pos, gamma) -> pos
        summary_rows.append({
            "T": T,
            "column": col,
            "n_peaks": len(detections_filtered),
            "positions": ", ".join(f"{p:.1f}" for p in positions),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["n_peaks_jump"] = summary_df["n_peaks"].diff().abs().fillna(0)

    return {
        "cfg": cfg,
        "temps": temps,
        "col_names": col_names,
        "spectra_resampled": spectra_resampled,
        "detections_per_T": detections_per_T,
        "summary": summary_df,
    }


# ----------------------------------------------------------------------
# 4a. Small-multiples grid: one panel per temperature
# ----------------------------------------------------------------------

def plot_detection_grid(results, xlim=(20, 600), ncols=3, save_path=None):
    cfg = results["cfg"]
    temps = results["temps"]
    spectra = results["spectra_resampled"]
    detections_per_T = results["detections_per_T"]

    n = len(temps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for i, (T, spec, detections) in enumerate(zip(reversed(temps), reversed(spectra), reversed(detections_per_T))):
        ax = axes[i]
        ax.plot(cfg.W, spec, color="black", lw=0.8)
        for (conf, A, pos, gamma) in detections:
            ax.axvline(pos, color="#e53e3e", ls="--", lw=0.8, alpha=0.8)
        ax.set_xlim(*xlim)
        ax.set_title(f"{T:g} K  (n={len(detections)})", fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("FCN detections (raw-signal filtered, no fitting) across temperature series",
                  fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved detection grid to {save_path}")
    return fig


# ----------------------------------------------------------------------
# 4b. Waterfall: all spectra stacked with an offset, peak markers overlaid
# ----------------------------------------------------------------------

def plot_detection_waterfall(results, xlim=(20, 600), offset_frac=1.15, save_path=None):
    cfg = results["cfg"]
    temps = results["temps"]
    spectra = results["spectra_resampled"]
    detections_per_T = results["detections_per_T"]

    # Normalize each spectrum's own peak height so the offset is legible
    # even if absolute intensity varies a lot across temperature.
    norm_heights = [np.nanmax(s[np.isfinite(s)]) if np.any(np.isfinite(s)) else 1.0
                     for s in spectra]
    step = np.nanmedian(norm_heights) * offset_frac if norm_heights else 1.0

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(temps) + 2))

    for i, (T, spec, detections) in enumerate(zip(temps, spectra, detections_per_T)):
        y_offset = i * step
        ax.plot(cfg.W, spec + y_offset, color="black", lw=0.7)
        for (conf, A, pos, gamma) in detections:
            idx = np.searchsorted(cfg.W, pos)
            idx = np.clip(idx, 0, len(spec) - 1)
            ax.plot(pos, spec[idx] + y_offset, marker="v", color="#e53e3e", ms=4, lw=0)
        ax.text(xlim[1], y_offset, f"  {T:g} K", va="center", fontsize=8)

    ax.set_xlim(*xlim)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_yticks([])
    ax.set_title("Detected-peak waterfall across temperature series (no fitting)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved waterfall plot to {save_path}")
    return fig


# ----------------------------------------------------------------------
# 5. CLI entry point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detection-only temperature series scan.")
    parser.add_argument("--csv", required=True, help="Wide-format T-series CSV.")
    parser.add_argument("--model", required=True, help="Path to dense_model.pt")
    parser.add_argument("--config", required=True, help="Path to dense_model_config.json")
    parser.add_argument("--out-dir", default="detection_out", help="Output directory.")
    parser.add_argument("--xlim", type=float, nargs=2, default=(20, 600),
                         help="X-axis range (cm^-1) for both plots.")
    parser.add_argument("--ncols", type=int, default=3, help="Columns in the small-multiples grid.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = run_detection_scan(args.csv, args.model, args.config)

    summary_path = os.path.join(args.out_dir, "detected_peaks_summary.csv")
    grid_path = os.path.join(args.out_dir, "detection_grid.pdf")
    waterfall_path = os.path.join(args.out_dir, "detection_waterfall.pdf")

    results["summary"].to_csv(summary_path, index=False)
    plot_detection_grid(results, xlim=tuple(args.xlim), ncols=args.ncols, save_path=grid_path)
    plot_detection_waterfall(results, xlim=tuple(args.xlim), save_path=waterfall_path)

    print("\n=== Peak-count jumps between adjacent temperatures ===")
    print(results["summary"][["T", "column", "n_peaks", "n_peaks_jump"]].to_string(index=False))


if __name__ == "__main__":
    main()
