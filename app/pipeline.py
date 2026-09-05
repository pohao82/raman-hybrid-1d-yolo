"""Framework-independent inference and fitting pipeline for Raman spectra.

No UI framework is imported here.  This module can be used from Streamlit,
Dash, notebooks, tests, or a future API service.
"""

import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from configs import Config
from processing.peak_filters_by_raw_signal import (
    filter_by_multiscale_movavg,
    filter_by_raw_signal_flat,
)
from processing.refinement_fit import refine, refine_grouped
from src.inference import (
    load_model,
    predict_peaks_dense,
    resample_to_training_grid,
)
from src.lineshapes import pseudo_voigt


def load_detector_model(
    model_path: str,
    config_path: str,
    device=None,
) -> Tuple[Any, Config, Any]:
    """Load the trained DenseDetector model in evaluation mode."""
    import torch

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    model, cfg, device = load_model(
        model_path,
        config_path,
        device=device,
        mode="eval",
    )
    return model, cfg, device


def parse_spectrum_file(
    file_obj,
    filename: str,
) -> Optional[pd.DataFrame]:
    """Parse CSV/TXT/TSV-like spectral data into a DataFrame."""
    try:
        content = file_obj.read()
        text = (
            content.decode("utf-8", errors="replace")
            if isinstance(content, bytes)
            else content
        )

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not lines:
            return pd.DataFrame()

        first_line = lines[0]
        comma_separated = "," in first_line
        sep = "," if comma_separated else r"\s+"

        tokens = (
            first_line.split(",")
            if comma_separated
            else first_line.split()
        )
        headerless = all(_is_float(token) for token in tokens)

        if headerless:
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                header=None,
                engine="python",
            )
            n_cols = df.shape[1]
            if n_cols == 2:
                df.columns = ["Wavenumber (cm⁻¹)", "Intensity"]
            elif n_cols > 2:
                df.columns = [
                    "Wavenumber (cm⁻¹)"
                ] + [
                    f"Intensity (Col {i})"
                    for i in range(1, n_cols)
                ]
            else:
                df.columns = ["Intensity"]
        else:
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                header=0,
                engine="python",
            )

        return df
    except Exception as exc:
        raise ValueError(
            f"Error parsing file '{filename}': {exc}"
        ) from exc


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def preprocess_spectrum(
    raw_freq: np.ndarray,
    raw_intensity: np.ndarray,
    cfg: Config,
    apply_savgol: bool = True,
    window_length: int = 5,
    polyorder: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare detection and fitting signals on the model's training grid."""
    smooth = raw_intensity.copy()

    if apply_savgol and len(raw_intensity) > 3:
        p = int(polyorder)
        w = int(window_length)

        if w % 2 == 0:
            w += 1
        if w <= p:
            w = p + 1 if (p + 1) % 2 else p + 2

        max_window = (
            len(raw_intensity)
            if len(raw_intensity) % 2
            else len(raw_intensity) - 1
        )
        w = min(w, max_window)
        smooth = savgol_filter(
            raw_intensity,
            window_length=w,
            polyorder=p,
        )

    return (
        resample_to_training_grid(raw_freq, smooth, cfg.W),
        resample_to_training_grid(raw_freq, raw_intensity, cfg.W),
    )


def run_fcn_detection(
    model: Any,
    resampled_signal: np.ndarray,
    cfg: Config,
    device: Any,
    conf_threshold: float = 0.50,
    use_movavg_filter: bool = False,
    idiff_threshold: float = 9.0,
    use_flat_filter: bool = True,
    flat_threshold: float = 0.10,
    resampled_raw_signal: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Run FCN inference followed by the optional noise filters."""
    raw_for_filter = (
        resampled_signal
        if resampled_raw_signal is None
        else resampled_raw_signal
    )

    detected_raw = predict_peaks_dense(
        model,
        resampled_signal,
        threshold=conf_threshold,
        cfg=cfg,
        device=device,
    )

    filtered = detected_raw

    if use_movavg_filter:
        filtered = filter_by_multiscale_movavg(
            filtered,
            cfg.W,
            raw_for_filter,
            cfg,
            threshold=idiff_threshold,
        )

    if use_flat_filter:
        filtered = filter_by_raw_signal_flat(
            filtered,
            cfg.W,
            resampled_signal,
            cfg,
            threshold=flat_threshold,
        )

    return detected_raw, filtered


def crop_active_fitting_window(
    resampled_raw_signal: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop leading/trailing zero regions before curve fitting."""
    nonzero_idx = np.flatnonzero(resampled_raw_signal)

    if len(nonzero_idx) == 0:
        return cfg.W, resampled_raw_signal

    start, end = nonzero_idx[0], nonzero_idx[-1]
    return cfg.W[start:end], resampled_raw_signal[start:end]


def run_refinement_pipeline(
    freq_in: np.ndarray,
    raw_signal_in: np.ndarray,
    peaks: List[Tuple],
    refine_mode: str = "Divide-and-Conquer Regional (`refine_grouped`)",
    pos_window: float = 0.5,
    width_scale: float = 1.5,
    target_peaks_per_grp: int = 5,
    sep_factor: float = 4.0,
    from_fitted: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run bounded non-linear least-squares Pseudo-Voigt refinement."""
    if not peaks:
        return None

    if "Regional" in refine_mode:
        return refine_grouped(
            freq_in,
            raw_signal_in,
            peaks,
            target_peaks_per_group=target_peaks_per_grp,
            separation_factor=sep_factor,
            pos_window=pos_window,
            width_scale=width_scale,
            from_fitted=from_fitted,
            verbose=0,
        )

    return refine(
        freq_in,
        raw_signal_in,
        peaks,
        pos_window=pos_window,
        width_scale=width_scale,
        from_fitted=from_fitted,
        verbose=0,
    )


def compute_fit_residual_and_rms(
    freq: np.ndarray,
    raw_signal: np.ndarray,
    fit_result: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """Compute point-wise residual and RMS for a refinement result."""
    refined_peaks = fit_result["peaks"]
    b0, b1 = fit_result["baseline"][:2]
    baseline = b0 + b1 * freq

    peak_signal = np.zeros_like(freq, dtype=float)
    for A, x0, sigma, eta in refined_peaks:
        peak_signal += pseudo_voigt(
            freq, A, x0, sigma, eta
        )

    reconstructed = baseline + peak_signal
    residual = raw_signal - reconstructed
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return residual, rms


def fitted_to_seeds(peaks_fitted: List[Tuple]) -> List[Tuple]:
    """Normalize edited fitted rows into valid refinement seeds."""
    seeds = []
    for row in peaks_fitted:
        A, x0, sigma, eta = (list(row) + [0.5])[:4]
        seeds.append(
            (
                float(A),
                float(x0),
                max(float(sigma), 1e-3),
                min(max(float(eta), 0.0), 1.0),
            )
        )
    return seeds


def build_peak_dataframe(
    refined_peaks: List[Tuple],
) -> pd.DataFrame:
    """Build the user-facing refined peak table."""
    rows = []

    for i, (A, x0, sigma, eta) in enumerate(refined_peaks):
        rows.append(
            {
                "Peak #": i + 1,
                "Center Position (cm⁻¹)": round(float(x0), 2),
                "Amplitude (Height)": round(float(A), 4),
                "FWHM Width (σ)": round(float(sigma), 3),
                "Mixing Fraction (η)": round(float(eta), 3),
                "Lorentzian Character": f"{int(round(float(eta) * 100))}%",
                "Gaussian Character": f"{int(round((1.0 - float(eta)) * 100))}%",
            }
        )

    return pd.DataFrame(rows)
