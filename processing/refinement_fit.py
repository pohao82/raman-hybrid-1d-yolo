"""
Refine FCN-predicted Raman peak parameters via classical nonlinear
least-squares, using pseudo-Voigt lineshapes instead of pure Lorentzians
(faster-decaying tails than the FCN's Lorentzian output).

The `baseline` key in every result dict is a fixed (0.0, 0.0) placeholder,
kept only so downstream plotting code keeps working.

Input:  freq (1D array), raw_signal (1D array),
        predicted peaks: list of (A, pos, gamma) from your FCN
Output: refined peak params + fitted curve + residual
"""

import numpy as np
from scipy.optimize import least_squares
from src.lineshapes import pseudo_voigt, voigt
from processing.peak_grouping import partition_peaks


# ----------------------------------------------------------------------
# Build the least_squares seed (x0) + bounds from a peak list
# ----------------------------------------------------------------------
def build_guess(peaks, pos_window=5.0, width_scale=3.0, from_fitted=False):
    """
    Flatten a peak list into the least_squares seed `x0` plus `(lb, ub)`
    bounds, laid out as
        [A1, x0_1, sigma1, eta1,  A2, x0_2, sigma2, eta2,  ...]

    Per peak the seed is (A, x0, sigma, eta) -- clamped feasible (A >= 0,
    sigma >= 1e-3, 0 <= eta <= 1) -- with bounds recentered on it:
    position +/- `pos_window`, width within a `width_scale` multiple,
    amplitude in [0, 3*A].

    from_fitted=False (default): `peaks` is FCN output (A, pos, gamma).
        The FCN gives no lineshape, so sigma0 = 2*gamma (pseudo-Voigt sigma
        ~ Lorentzian FWHM) and eta0 = 0.5.
    from_fitted=True: `peaks` is an ALREADY-refined [A, x0, sigma, eta]
        list; every parameter -- eta included -- resumes at its fitted
        value, so re-fitting an unedited list is a near no-op (the
        optimizer starts at the minimum and exits immediately).
    """
    x0_list, lb_list, ub_list = [], [], []

    for p in peaks:
        if from_fitted:
            A, x0, sigma, eta = p
        # initial from FCN
        else:
            A, x0, gamma = p
            sigma, eta = 2.0 * max(float(gamma), 1e-3), 0.5

        A = max(float(A), 0.0)
        x0 = float(x0)
        sigma = max(float(sigma), 1e-3)
        eta = min(max(float(eta), 0.0), 1.0)

        x0_list += [A, x0, sigma, eta]
        lb_list += [0.0, x0 - pos_window, sigma / width_scale, 0.0]
        ub_list += [3.0 * max(A, 1e-6), x0 + pos_window, sigma * width_scale, 1.0]

    return np.array(x0_list), (np.array(lb_list), np.array(ub_list))


# ----------------------------------------------------------------------
# Composite model: sum of peaks only -- no baseline term (see module
# docstring). Any background must be subtracted by the caller beforehand.
# ----------------------------------------------------------------------
PARAMS_PER_PEAK = 4  # A, x0, sigma, eta  (pseudo-voigt)


# The optimizer only understands flat 1D arrays.
# build_guess flattens the peak parameters in to the following form
# params = [A1, x0_1, sigma1, eta1,  A2, x0_2, sigma2, eta2,  A3, x0_3, sigma3, eta3, ...]
def unpack(params, n_peaks):
    # reshape the flat params back into n_peaks-by-PARAMS_PER_PEAK
    return params.reshape(n_peaks, PARAMS_PER_PEAK)


# Reconstruction through parameters
def model(params, x, n_peaks):
    peaks = unpack(params, n_peaks)
    y = np.zeros_like(x, dtype=float)
    for A, x0, sigma, eta in peaks:
        y += pseudo_voigt(x, A, x0, sigma, eta)
    return y


def residuals(params, x, y, n_peaks):
    return model(params, x, n_peaks) - y


def _fit_peaks(freq, target, x0, bounds, loss="soft_l1", f_scale=None, verbose=0):
    """
    Core peaks-only nonlinear least-squares fit: given a flat seed `x0` and
    `bounds` (see `build_guess`), fit the pseudo-Voigt sum against `target`.
    No baseline term at all -- any background must already have been
    subtracted from `target` by the caller.
    """
    freq = np.asarray(freq, dtype=float)
    target = np.asarray(target, dtype=float)
    n_peaks = len(x0) // PARAMS_PER_PEAK

    if f_scale is None:
        peak_amp = np.nanmax(np.abs(target)) if target.size else 1.0
        f_scale = 0.05 * peak_amp if peak_amp > 0 else 1.0

    result = least_squares(
        residuals, x0, bounds=bounds,
        args=(freq, target, n_peaks),
        loss=loss,          # robust loss softens influence of tail mismatch
        f_scale=f_scale,
        method="trf",
        x_scale="jac",      # auto-scale params (A~1, x0~500, sigma~5, eta~0.5)
        verbose=verbose,
    )

    peaks_refined = unpack(result.x, n_peaks)
    fitted_curve = model(result.x, freq, n_peaks)

    return {
        "success": result.success,
        "params": result.x,
        "peaks": peaks_refined,      # array of [A, x0, sigma, eta] per peak
        "fitted_curve": fitted_curve,
        "residual": target - fitted_curve,
        "cost": result.cost,
        "raw_result": result,
    }


# ----------------------------------------------------------------------
# Public entry points: refine() is the single-group wrapper; the core
# implementation is refine_grouped() further down.
# ----------------------------------------------------------------------
def refine(freq, raw_signal, predicted_peaks, pos_window=5.0,
           width_scale=3.0, loss="soft_l1", f_scale=None, verbose=1,
           from_fitted=False):
    """
    Thin wrapper -- does `refine_grouped(..., single_group=True)`
    """
    return refine_grouped(
        freq, raw_signal, predicted_peaks,
        pos_window=pos_window, width_scale=width_scale,
        loss=loss, f_scale=f_scale, verbose=verbose,
        from_fitted=from_fitted, single_group=True,
    )


def _group_fit_window(freq, groups, boundaries, g_idx, pad):
    """Boolean mask over `freq` for group `g_idx`'s fit window: its core
    region [boundaries[g_idx], boundaries[g_idx + 1]] widened by `pad` on
    each side for tail context (only the core is stitched into the output,
    so neighboring groups never double-count).

    Each side's `pad` is capped at half the gap to the nearest neighboring
    peak, so the window can't swallow a neighbor's signal that this group's
    peaks don't model -- which the fit could only absorb by distorting its
    own amplitude/width.
    """
    core_lo, core_hi = boundaries[g_idx], boundaries[g_idx + 1]
    local_pad = pad
    if g_idx > 0:
        local_pad = min(local_pad, 0.5 * max(core_lo - groups[g_idx - 1][-1][1], 0.0))
    if g_idx < len(groups) - 1:
        local_pad = min(local_pad, 0.5 * max(groups[g_idx + 1][0][1] - core_hi, 0.0))
    local_pad = max(local_pad, 0.0)
    return (freq >= core_lo - local_pad) & (freq <= core_hi + local_pad)


def refine_grouped(freq, raw_signal, predicted_peaks, target_peaks_per_group=5,
                   separation_factor=3.0, max_group_multiple=20, pad=15.0,
                   pos_window=5.0, width_scale=3.0,
                   loss="soft_l1", f_scale=None, verbose=0, from_fitted=False,
                   single_group=False):
    """
    Core refinement routine. Partitions predicted_peaks into groups (see
    `partition_peaks`), fits each group's peaks independently over a
    locally padded window, then stitches the results into one full-length
    curve. Preferred over one all-at-once fit when you have many peaks
    (25+): smaller subproblems, less cross-talk between distant peaks.

    `single_group=True` puts every peak in one group spanning the whole
    spectrum -- i.e. the global simultaneous fit exposed as `refine()`.

    `raw_signal` is fit as-is -- no background is ever estimated or
    removed here. If your spectrum has a background, subtract it BEFORE
    calling (see `processing.baseline_fit`).

        pad: extra x-units of fit context on each side of a group's core
             region, neighbor-capped (see `_group_fit_window`). Only the
             core region is stitched into the output, so neighboring
             group fits never double-count.
        from_fitted: if True, `predicted_peaks` is an already-refined
             [A, x0, sigma, eta] list and each group resumes from those
             values (see `build_guess`) instead of restarting from FCN
             (A, pos, gamma). Re-fitting an unedited list then converges
             immediately.

    Returns a dict:
        success:       True iff every group's least-squares converged
        cost:          summed least-squares cost across groups
        peaks:         array of [A, x0, sigma, eta] stacked across all
                       groups, in left-to-right (sorted position) order
        baseline:      always (0.0, 0.0) -- a placeholder kept so
                       plotting code that does
                       `b0, b1 = fit_result["baseline"]` keeps working.
        fitted_curve:  full-length stitched model curve (peaks only)
        residual:      raw_signal - fitted_curve
        boundaries:    group boundary frequencies (see partition_peaks)
        groups:        the partitioned predicted_peaks, per group
        group_results: list of per-group dicts (peaks-only fit output:
                       params, peaks, success, cost, raw_result, ...)
                       plus its (core_lo, core_hi) boundary -- useful
                       for inspecting/debugging individual group fits.
                       For `single_group=True` / `refine()` this has one
                       entry holding the whole global fit.
    """
    freq = np.asarray(freq, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)

    # from_fitted peaks carry sigma (~2*gamma) in slot [2] instead of the
    # FCN gamma partition_peaks expects, so halve separation_factor to keep
    # the island cuts at the same physical scale as the initial fit.
    part_sep = separation_factor / 2.0 if from_fitted else separation_factor
    groups, boundaries = partition_peaks(
        freq, raw_signal, predicted_peaks,
        target_peaks_per_group=target_peaks_per_group,
        separation_factor=part_sep,
        max_group_multiple=max_group_multiple,
        single_group=single_group,
    )

    fitted_peaks_only = np.zeros_like(raw_signal)
    all_peaks = []
    group_results = []

    for g_idx, peaks_in_group in enumerate(groups):
        core_lo, core_hi = boundaries[g_idx], boundaries[g_idx + 1]

        # Compare user specified pad and group boundary and estimate masked area
        fit_mask = _group_fit_window(freq, groups, boundaries, g_idx, pad)
        if not np.any(fit_mask):
            continue

        sub_freq = freq[fit_mask]
        sub_target = raw_signal[fit_mask]

        x0, bounds = build_guess(peaks_in_group, pos_window, width_scale, from_fitted)
        result = _fit_peaks(sub_freq, sub_target, x0, bounds,
                            loss=loss, f_scale=f_scale, verbose=verbose)

        # Add contribution to total signal
        fitted_peaks_only += model(
            result["params"], freq, len(peaks_in_group))

        # Add peaks params to total list
        all_peaks.append(result["peaks"])
        group_results.append({
            "core_bounds": (core_lo, core_hi),
            "n_peaks": len(peaks_in_group),
            **result,
        })

    combined_peaks = (np.vstack(all_peaks) if all_peaks
                      else np.empty((0, PARAMS_PER_PEAK)))
    fitted_curve = fitted_peaks_only
    residual = raw_signal - fitted_curve

    return {
        "success": all(g["success"] for g in group_results) if group_results else True,
        "cost": float(sum(g["cost"] for g in group_results)),
        "peaks": combined_peaks,
        "baseline": (0.0, 0.0),   # placeholder -- baseline handled by the caller
        "fitted_curve": fitted_curve,
        "residual": residual,
        "boundaries": boundaries,
        "groups": groups,
        "group_results": group_results,
    }



# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
#if __name__ == "__main__":
#    # --- fake demo data, replace with your freq / raw_signal / FCN output ---
#    rng = np.random.default_rng(0)
#    freq = np.linspace(200, 1800, 2000)
#    true_peaks = [(1.0, 520.0, 6.0), (0.6, 950.0, 10.0), (0.8, 1350.0, 8.0)]
#    raw_signal = sum(pseudo_voigt(freq, A, p, 2 * g, 0.5) for A, p, g in true_peaks)
#    raw_signal += 0.01 * rng.normal(size=freq.size)
#
#    # predicted_peaks = fcn_model(freq, raw_signal)  # <- your model's output
#    predicted_peaks = [(0.9, 522.0, 5.0), (0.55, 946.0, 9.0), (0.7, 1355.0, 7.5)]
#
#    out = refine(freq, raw_signal, predicted_peaks)
#
#    print("\nRefined peaks (A, x0, sigma, eta):")
#    for row in out["peaks"]:
#        print(row)
#    print("Residual RMS:", np.sqrt(np.mean(out["residual"] ** 2)))
#
#    # refine(...) == refine_grouped(..., single_group=True). For many peaks,
#    # drop single_group and let it partition:
#    # out = refine_grouped(
#    #     freq, raw_signal, predicted_peaks,
#    #     target_peaks_per_group=5,    # soft target group size
#    #     separation_factor=3.0,       # peaks must be this many widths apart to cut between them
#    #     pad=15.0,                    # extra x-units of context around each group when fitting
#    # )
#    # print("n groups:", len(out["groups"]))
