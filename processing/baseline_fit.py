"""
Standalone linear/polynomial background estimation for Raman spectra.

Decoupled from peak fitting on purpose: background estimation here is a
closed-form masked polyfit (fast, deterministic given freq/raw_signal/peaks),
entirely separate from the nonlinear least-squares that fits peak lineshapes
in `processing.refinement_fit`. Those `refine()` / `refine_grouped()`
routines never touch a background -- if a spectrum has one, subtract it here
first and pass the flattened signal in:

    coeffs = estimate_global_baseline(freq, raw, predicted_peaks)
    flat = raw - eval_baseline(coeffs, freq)
    result = refine(freq, flat, predicted_peaks)
"""

import numpy as np


# Experimental, not reliable
def estimate_global_baseline(freq, raw_signal, predicted_peaks, degree=1,
                              window_factor=3.0, n_iter=3, clip_sigma=3.0):
    """
    Fit ONE polynomial baseline (degree `degree`) for the whole
    spectrum, ONLY using the samples that fall outside
    `window_factor * width` of every predicted peak. A few rounds of
    sigma-clipping reject any leftover peak shoulders or outliers that
    slip into the baseline-only mask.

    A single shared baseline, fit once over the whole spectrum (away from
    every peak) and subtracted before any peak fitting happens, keeps the
    reconstructed background continuous -- unlike per-window baselines,
    which are weakly constrained and disagree with their neighbors, leaving
    a patchy background once independent fits are stitched together.

        predicted_peaks: (A, pos, gamma) FCN peaks or (A, x0, sigma, eta)
                already-refined peaks -- either shape works, since only
                index 1 (position) and index 2 (a width-like parameter)
                are used to build the exclusion mask.
        degree: polynomial order. 1 = linear (b0 + b1*x). Raise it if
                your background genuinely curves (e.g. broad Raman
                fluorescence humps) -- but since the fit only uses
                sparse, masked-out points and is unconstrained outside
                that support, keep it modest (2-3) unless you have a
                specific reason to go higher; too-high degree on
                sparse/clustered baseline points can ring or diverge
                near the spectrum edges (Runge's phenomenon).

    Returns a tuple of `degree + 1` coefficients, ascending order
    (c0, c1, c2, ...) such that baseline(x) = sum(c_k * x**k); pair with
    `eval_baseline`. Falls back to all-zero coefficients if no
    baseline-only samples are found (e.g. peaks cover the whole spectrum).
    """
    freq = np.asarray(freq, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)

    mask = np.ones_like(freq, dtype=bool)
    for p in predicted_peaks:
        pos, width = float(p[1]), max(float(p[2]), 1e-3)
        mask &= ~((freq >= pos - window_factor * width) &
                  (freq <= pos + window_factor * width))

    if not np.any(mask) or mask.sum() <= degree:
        return tuple([0.0] * (degree + 1))

    bx, by = freq[mask], raw_signal[mask]
    coeffs = np.polyfit(bx, by, degree)  # highest degree first

    for _ in range(n_iter):
        resid = by - np.polyval(coeffs, bx)
        std = np.std(resid)
        if std == 0:
            break
        keep = np.abs(resid) <= clip_sigma * std
        if keep.sum() <= degree or keep.all():
            break
        bx, by = bx[keep], by[keep]
        coeffs = np.polyfit(bx, by, degree)

    return tuple(float(c) for c in coeffs[::-1])  # ascending order: (c0, c1, ...)


def eval_baseline(coeffs, x):
    """Evaluate a baseline given ascending-order coefficients (c0, c1, ...)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for k, c in enumerate(coeffs):
        y = y + c * x ** k
    return y
