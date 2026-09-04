"""
Inference and Data Processing Pipeline for Raman Peak Detection & Refinement.

Decouples numerical processing, model inference, and curve fitting algorithms
from the Streamlit UI presentation layer.
"""

import os
import io
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from scipy.signal import savgol_filter

from configs import Config
from src.lineshapes import pseudo_voigt
from src.inference import load_model, predict_peaks_dense, resample_to_training_grid
from processing.peak_filters_by_raw_signal import filter_by_raw_signal_flat, filter_by_multiscale_movavg
from processing.refinement_fit import refine, refine_grouped


def load_detector_model(model_path: str, config_path: str, device=None) -> Tuple[Any, Config, Any]:
    """
    Loads the trained DenseDetector PyTorch model and its associated Config.
    """
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    model, cfg, device = load_model(model_path, config_path, device=device, mode="eval")
    return model, cfg, device


def parse_spectrum_file(file_obj, filename: str) -> Optional[pd.DataFrame]:
    """
    Parses uploaded CSV/TXT/TSV spectrum data into a pandas DataFrame.
    Automatically detects whether the file is headerless or contains column headers.
    """
    try:
        content = file_obj.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = content

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not lines:
            return pd.DataFrame()

        first_line = lines[0]
        sep = "," if "," in first_line else r"\s+"

        # Check if the first line is completely numeric (headerless data)
        tokens = first_line.split(",") if "," in first_line else first_line.split()
        is_headerless = True
        for token in tokens:
            try:
                float(token)
            except ValueError:
                is_headerless = False
                break

        if is_headerless:
            df = pd.read_csv(io.StringIO(text), sep=sep, header=None, engine="python")
            n_cols = df.shape[1]
            if n_cols == 2:
                df.columns = ["Wavenumber (cm⁻¹)", "Intensity"]
            elif n_cols > 2:
                df.columns = ["Wavenumber (cm⁻¹)"] + [f"Intensity (Col {i})" for i in range(1, n_cols)]
            else:
                df.columns = ["Intensity"]
        else:
            df = pd.read_csv(io.StringIO(text), sep=sep, header=0, engine="python")

        return df
    except Exception as e:
        raise ValueError(f"Error parsing file '{filename}': {e}")


def preprocess_spectrum(
    raw_freq: np.ndarray,
    raw_intensity: np.ndarray,
    cfg: Config,
    apply_savgol: bool = True,
    window_length: int = 5,
    polyorder: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the raw experimental spectrum.

    1. Pre-smooths raw intensity (for FCN detection only).
    2. Resamples smoothed intensity onto cfg.W (for FCN input).
    3. Resamples raw intensity onto cfg.W (for unbiased curve fitting).

    Returns
    -------
    resampled_smoothed : np.ndarray
        Signal used for FCN peak detection.
    resampled_raw : np.ndarray
        Unaltered signal used for physical curve fitting.
    """
    smooth_raw_intensity = raw_intensity.copy()
    if apply_savgol and len(raw_intensity) > 3:
        p = int(polyorder)
        w = int(window_length)
        if w % 2 == 0:
            w += 1
        if w <= p:
            w = p + 1 if (p + 1) % 2 == 1 else p + 2
        w = min(w, len(raw_intensity) if len(raw_intensity) % 2 == 1 else len(raw_intensity) - 1)
        smooth_raw_intensity = savgol_filter(raw_intensity, window_length=w, polyorder=p)

    resampled_smoothed = resample_to_training_grid(raw_freq, smooth_raw_intensity, cfg.W)
    resampled_raw = resample_to_training_grid(raw_freq, raw_intensity, cfg.W)
    return resampled_smoothed, resampled_raw


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
    """
    Executes forward FCN inference and (optionally chained) detection filtering.

    The two filters are independent on/off toggles applied in sequence:
    multiscale moving-average i_diff first, then flat intensity threshold. The
    multiscale filter needs the UNSMOOTHED signal for its noise-sigma estimate --
    pass `resampled_raw_signal`; it falls back to `resampled_signal` otherwise.

    Returns
    -------
    detected_raw : list of (conf, A, pos, gamma)
        Raw detections from the FCN.
    filtered : list of (conf, A, pos, gamma)
        `detected_raw` passed through whichever filters are enabled (score always
        preserved). With no filter enabled, this is `detected_raw` unchanged.
    """
    raw_for_filter = resampled_signal if resampled_raw_signal is None else resampled_raw_signal
    detected_raw = predict_peaks_dense(model, resampled_signal, threshold=conf_threshold, cfg=cfg, device=device)

    filtered = detected_raw
    if use_movavg_filter:
        filtered = filter_by_multiscale_movavg(filtered, cfg.W, raw_for_filter, cfg, threshold=idiff_threshold)
    if use_flat_filter:
        filtered = filter_by_raw_signal_flat(filtered, cfg.W, resampled_signal, cfg, threshold=flat_threshold)

    return detected_raw, filtered


def crop_active_fitting_window(
    resampled_raw_signal: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops the active non-zero spectral range for curve fitting.
    """
    nonzero_idx = np.flatnonzero(resampled_raw_signal)
    if len(nonzero_idx) > 0:
        f_begin, f_end = nonzero_idx[0], nonzero_idx[-1]
        freq_in = cfg.W[f_begin:f_end]
        raw_signal_in = resampled_raw_signal[f_begin:f_end]
    else:
        freq_in = cfg.W
        raw_signal_in = resampled_raw_signal
    return freq_in, raw_signal_in


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
    resume_baseline: Tuple[float, float] = (0.0, 0.0),
) -> Optional[Dict[str, Any]]:
    """
    Executes bounded non-linear least squares Pseudo-Voigt fitting.

    `peaks` is normally (A, pos, gamma) -- strip the score first if you have
    detections (conf, A, pos, gamma).

    With `from_fitted=True`, `peaks` is an already-refined
    (A, x0, sigma, eta) list and the fit resumes from it (see
    `build_guess_from_fitted`): re-fitting an unedited list converges
    immediately instead of restarting from FCN-shaped assumptions.
    """
    if len(peaks) == 0:
        return None

    if "Regional" in refine_mode:
        fit_result = refine_grouped(
            freq_in, raw_signal_in, peaks,
            target_peaks_per_group=target_peaks_per_grp,
            separation_factor=sep_factor,
            pos_window=pos_window,
            width_scale=width_scale,
            from_fitted=from_fitted,
            verbose=0
        )
    else:
        fit_result = refine(
            freq_in, raw_signal_in, peaks,
            pos_window=pos_window,
            width_scale=width_scale,
            from_fitted=from_fitted,
            resume_baseline=resume_baseline,
            verbose=0
        )
    return fit_result


def compute_fit_residual_and_rms(
    freq: np.ndarray,
    raw_signal: np.ndarray,
    fit_result: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """
    Calculates point-by-point residual and root-mean-square (RMS) error.
    """
    refined_peaks = fit_result["peaks"]  # (A, x0, sigma, eta) each
    b0, b1 = fit_result["baseline"][:2]
    baseline = b0 + b1 * freq

    optimized_peaks_signal = np.zeros_like(freq, dtype=float)
    for A, x0, sigma, eta in refined_peaks:
        optimized_peaks_signal += pseudo_voigt(freq, A, x0, sigma, eta)
    reconstructed_fit = baseline + optimized_peaks_signal

    residual = raw_signal - reconstructed_fit
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return residual, rms


def fitted_to_seeds(peaks_fitted: List[Tuple]) -> List[Tuple]:
    """Normalize edited fitted rows into resume seeds (A, x0, sigma, eta) for
    `run_refinement_pipeline(..., from_fitted=True)`. All four parameters,
    eta included, are carried through so a re-fit starts at the converged
    state rather than restarting from sigma0 = 2*gamma / eta0 = 0.5."""
    seeds = []
    for row in peaks_fitted:
        A, x0, sigma, eta = (list(row) + [0.5])[:4]
        seeds.append((float(A), float(x0), max(float(sigma), 1e-3),
                      min(max(float(eta), 0.0), 1.0)))
    return seeds


def build_peak_dataframe(refined_peaks: List[Tuple]) -> pd.DataFrame:
    """
    Formats refined peaks (A, x0, sigma, eta) into a display / export DataFrame.
    """
    table_rows = []
    for i, (A, x0, sigma, eta) in enumerate(refined_peaks):
        table_rows.append({
            "Peak #": i + 1,
            "Center Position (cm⁻¹)": round(float(x0), 2),
            "Amplitude (Height)": round(float(A), 4),
            "FWHM Width (σ)": round(float(sigma), 3),
            "Mixing Fraction (η)": round(float(eta), 3),
            "Lorentzian Character": f"{int(round(float(eta)*100))}%",
            "Gaussian Character": f"{int(round((1.0 - float(eta))*100))}%"
        })
    return pd.DataFrame(table_rows)
