"""
Standardized & Unified Plotting Module for Raman Spectrum Reconstruction.

Provides `plot_reconstruction` and `plot_raw_spectrum` to visualize:
  - Raw experimental measurements vs resampled signal.
  - Reconstructed multi-peak fits from optimized Pseudo-Voigt parameters (after fitting).
  - Decomposed individual lineshape components with background shading.
  - Initial FCN proxy reconstructions (before fitting).
  - Vertical reference markers for candidate peak positions.
  - Residual error panels underneath.

Designed to provide a single, unified plotting backend shared across
`app.py` (Streamlit GUI) and Jupyter notebooks (`notebooks/`).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any, Union

from src.lineshapes import pseudo_voigt, lorentzian_np


def _normalize_predicted_peaks(predicted_peaks: Optional[List[Union[Tuple, List]]]) -> List[Tuple[float, float, float]]:
    """Normalizes predicted peaks to [(A, pos, gamma), ...]."""
    if predicted_peaks is None:
        return []
    clean = []
    for p in predicted_peaks:
        if len(p) == 4:
            # (conf, A, pos, gamma)
            clean.append((float(p[1]), float(p[2]), float(p[3])))
        elif len(p) == 3:
            # (A, pos, gamma)
            clean.append((float(p[0]), float(p[1]), float(p[2])))
    return clean


def plot_reconstruction(
    freq: np.ndarray,
    raw_signal: np.ndarray,
    fit_result: Optional[Dict[str, Any]] = None,
    predicted_peaks: Optional[List[Union[Tuple, List]]] = None,
    raw_freq: Optional[np.ndarray] = None,
    raw_intensity: Optional[np.ndarray] = None,
    show_optimized_fit: bool = True,
    show_components: bool = True,
    show_initial_proxy: bool = False,
    show_predicted_lines: bool = True,
    title: str = "Raman Spectrum Fit",
    xlabel: str = "Raman shift (cm$^{-1}$)",
    ylabel: str = "Intensity (a.u.)",
    xlim: Optional[Tuple[float, float]] = None,
    ylim_main: Optional[Tuple[float, float]] = None,
    ylim_res: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 200,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    **kwargs: Any,
) -> plt.Figure:
    """
    Standardized, unified plotting function for Raman spectrum analysis.

    Parameters
    ----------
    freq : np.ndarray
        Frequency / wavenumber axis for the resampled signal and fit.
    raw_signal : np.ndarray
        Unaltered resampled experimental signal on `freq`.
    fit_result : dict, optional
        Dictionary returned by `refine()` or `refine_grouped()`.
        Must contain 'peaks' [(A, x0, sigma, eta), ...] and 'baseline' (b0, b1).
        If None, generates a single-panel experimental spectrum view.
    predicted_peaks : list of tuples, optional
        Candidate peak detections from FCN (either (A, pos, gamma) or (conf, A, pos, gamma)).
    raw_freq : np.ndarray, optional
        Original un-resampled experimental frequency axis.
    raw_intensity : np.ndarray, optional
        Original un-resampled experimental intensity values.
    show_optimized_fit : bool, default True
        Whether to plot the reconstructed curve from optimized Pseudo-Voigt parameters (after fitting).
    show_components : bool, default True
        Whether to plot individual shaded Pseudo-Voigt lineshapes (after fitting).
    show_initial_proxy : bool, default False
        Whether to plot the initial Lorentzian proxy curve from FCN predictions (before fitting).
    show_predicted_lines : bool, default True
        Whether to draw vertical dotted lines at FCN candidate peak positions.
    title : str, default "Raman Spectrum Fit"
        Title for the top panel.
    xlabel : str, default "Raman shift (cm$^{-1}$)"
        Label for the x-axis.
    ylabel : str, default "Intensity (a.u.)"
        Label for the y-axis.
    xlim : tuple of (float, float), optional
        Custom x-axis limits (xmin, xmax).
    ylim_main : tuple of (float, float), optional
        Custom y-axis limits (ymin, ymax) for the main spectrum panel.
    ylim_res : tuple of (float, float), optional
        Custom y-axis limits (ymin, ymax) for the residual panel.
    figsize : tuple of (int, int), default (16, 8)
        Matplotlib figure dimensions.
    dpi : int, default 200
        Dots per inch (resolution) for the figure canvas rendering.
    save_path : str, optional
        If provided, saves the figure to the specified file path.
    show_plot : bool, default False
        Whether to call `plt.show()` (useful in notebooks/scripts).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    freq = np.asarray(freq, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)
    clean_predicted = _normalize_predicted_peaks(predicted_peaks)

    # -------------------------------------------------------------------------
    # CASE 1: Raw / Unfitted View (Single Panel)
    # -------------------------------------------------------------------------
    if fit_result is None:
        fig, ax = plt.subplots(figsize=(figsize[0], max(5, figsize[1] - 2)), dpi=dpi)
        
        # Raw ungridded data
        if raw_freq is not None and raw_intensity is not None:
            ax.plot(raw_freq, raw_intensity, color="black", lw=1.0, ls="--", alpha=0.85, label="Raw Experimental Data")
            ax.plot(freq, raw_signal, color="#4B5563", lw=0.9, alpha=0.7, label="Resampled Signal (on Model Grid)")
        else:
            ax.plot(freq, raw_signal, color="black", lw=1.0, label="Experimental Signal")

        # Optional: Initial FCN Proxy (Before Fitting)
        if show_initial_proxy and len(clean_predicted) > 0:
            proxy_before = np.zeros_like(freq, dtype=float)
            for A, pos, gamma in clean_predicted:
                proxy_before += lorentzian_np(freq, A, pos, gamma)
            ax.plot(freq, proxy_before, color="#2563EB", lw=1.3, ls="--", label="Initial FCN Proxy (Before Fitting)")

        # Candidate peak vertical reference lines
        if show_predicted_lines and len(clean_predicted) > 0:
            for A, pos, gamma in clean_predicted:
                ax.axvline(pos, color="steelblue", lw=1.0, ls=":", alpha=0.8)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim((np.min(freq), np.max(freq)))

        if ylim_main is not None:
            ax.set_ylim(ylim_main)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=max(dpi, 300))
        if show_plot:
            plt.show()

        return fig

    # -------------------------------------------------------------------------
    # CASE 2: Fitted View (Two Panels: Reconstruction + Residuals)
    # -------------------------------------------------------------------------
    peaks = fit_result.get("peaks", [])
    baseline_coeffs = fit_result.get("baseline", (0.0, 0.0))
    b0, b1 = float(baseline_coeffs[0]), float(baseline_coeffs[1])
    baseline = b0 + b1 * freq

    # Calculate reconstructed signal from optimized Pseudo-Voigt peaks (AFTER fitting)
    optimized_peaks_curve = np.zeros_like(freq, dtype=float)
    for A, x0, sigma, eta in peaks:
        optimized_peaks_curve += pseudo_voigt(freq, A, x0, sigma, eta)
    reconstructed_fit = baseline + optimized_peaks_curve

    residual = raw_signal - reconstructed_fit

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        dpi=dpi
    )

    # --- Main Panel: Raw Data, Fits, Baseline, Components ---
    if raw_freq is not None and raw_intensity is not None:
        ax1.plot(raw_freq, raw_intensity, color="black", lw=1.0, ls="--", alpha=0.8, label="Raw Experimental Data")
    else:
        ax1.plot(freq, raw_signal, color="black", lw=1.0, ls="--", label="Raw Signal")

    # Optional: Initial FCN Proxy (Before Fitting)
    if show_initial_proxy and len(clean_predicted) > 0:
        proxy_before = np.zeros_like(freq, dtype=float)
        for A, pos, gamma in clean_predicted:
            proxy_before += lorentzian_np(freq, A, pos, gamma)
        ax1.plot(freq, proxy_before, color="#2563EB", lw=1.3, ls="--", label="Initial FCN Proxy (Before Fitting)")

    # Reconstructed signal from optimized peaks (AFTER fitting)
    if show_optimized_fit:
        ax1.plot(freq, reconstructed_fit, color="crimson", lw=1.8, label="Optimized Fit (After Fitting)")

    ax1.plot(freq, baseline, color="gray", lw=1.0, ls=":", label="Linear Baseline")

    # Optional: Individual Pseudo-Voigt components (AFTER fitting)
    if show_components and len(peaks) > 0:
        for i, (A, x0, sigma, eta) in enumerate(peaks):
            comp = baseline + pseudo_voigt(freq, A, x0, sigma, eta)
            ax1.plot(freq, comp, lw=1.0, alpha=0.75, label=f"P{i+1}: {x0:.1f} cm⁻¹ (η={eta:.2f})")
            ax1.fill_between(freq, baseline, comp, alpha=0.08)

    # Candidate peak vertical reference lines
    if show_predicted_lines and len(clean_predicted) > 0:
        for A, pos, gamma in clean_predicted:
            ax1.axvline(pos, color="steelblue", lw=0.9, ls=":", alpha=0.7)

    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ncol = min(4, max(2, len(peaks) // 4 + 1)) if len(peaks) > 0 else 2
    ax1.legend(fontsize=8, ncol=ncol, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.3)

    if xlim is not None:
        ax1.set_xlim(xlim)
    else:
        ax1.set_xlim((np.min(freq), np.max(freq)))

    if ylim_main is not None:
        ax1.set_ylim(ylim_main)

    # --- Residual Panel ---
    ax2.plot(freq, residual, color="black", lw=0.9, label="Residual (Raw - Fit)")
    ax2.axhline(0, color="crimson", lw=0.8, ls="--")
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel("Residual", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)

    if xlim is not None:
        ax2.set_xlim(xlim)
    else:
        ax2.set_xlim((np.min(freq), np.max(freq)))

    if ylim_res is not None:
        ax2.set_ylim(ylim_res)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    if show_plot:
        plt.show()

    return fig
