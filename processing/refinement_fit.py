"""
Refine FCN-predicted Raman peak parameters via classical nonlinear
least-squares, using pseudo-Voigt lineshapes instead of pure Lorentzians
(faster-decaying tails than the FCN's Lorentzian output).

Input:  freq (1D array), raw_signal (1D array),
        predicted peaks: list of (A, pos, gamma) from your FCN
Output: refined peak params + fitted curve + residual
"""

import numpy as np
from scipy.optimize import least_squares
from src.lineshapes import pseudo_voigt, voigt


# ----------------------------------------------------------------------
# Build initial guess + bounds from FCN predictions
# ----------------------------------------------------------------------
def build_initial_guess(predicted_peaks, pos_window=5.0, width_scale=3.0):
    """
    This function flattens the peak parameter list in to x0 of the following form
    [A1, x0_1, sigma1, eta1,  A2, x0_2, sigma2, eta2,  A3, x0_3, sigma3, eta3, ...,  b0, b1]

        predicted_peaks: list of (A, pos, gamma) from the FCN
        pos_window: allowed +/- shift in x-units around FCN-predicted position
        width_scale: allowed multiplicative range for width around 2*gamma

    """
    x0_list, lb_list, ub_list = [], [], []

    # from detected peakks (A, x0, sigma, eta)
    for A, pos, gamma in predicted_peaks:
        gamma = max(gamma, 1e-3)
        sigma0 = 2.0 * gamma  # pseudo-voigt sigma ~ Lorentzian FWHM
        eta0 = 0.5

        x0_list += [A, pos, sigma0, eta0]
        lb_list += [
            0.0,                      # A >= 0
            pos - pos_window,         # position lower bound
            sigma0 / width_scale,     # width lower bound
            0.0,                      # eta lower bound
        ]
        ub_list += [
            3.0 * max(A, 1e-6),       # A upper bound
            pos + pos_window,         # position upper bound
            sigma0 * width_scale,     # width upper bound
            1.0,                      # eta upper bound
        ]

    # baseline: b0, b1 (linear). Loosen/tighten as needed.
    x0_list += [0.0, 0.0]
    lb_list += [-np.inf, -np.inf]
    ub_list += [np.inf, np.inf]

    return np.array(x0_list), (np.array(lb_list), np.array(ub_list))


def build_guess_from_fitted(fitted_peaks, pos_window=5.0, width_scale=3.0,
                            baseline=(0.0, 0.0)):
    """
    Same flat layout as `build_initial_guess`, but seeded from an ALREADY
    refined peak list -- [A, x0, sigma, eta] per peak -- so a re-fit resumes
    from the converged state instead of restarting from FCN-shaped assumptions
    (sigma0 = 2*gamma, eta0 = 0.5).

    Every parameter -- eta and the linear baseline (b0, b1) included -- starts
    at its fitted value; bounds are recentered on that value (position
    +/- pos_window, width within a width_scale multiple). Re-fitting an
    unedited list is then a near no-op: the optimizer starts at the minimum --
    including any eta pinned to 0 or 1 -- and exits immediately.

        fitted_peaks: iterable of (A, x0, sigma, eta)
        pos_window:   allowed +/- shift around each fitted position
        width_scale:  allowed multiplicative range around each fitted sigma
        baseline:     (b0, b1) from the previous fit -- carried forward so the
                      optimizer doesn't have to re-discover it from zero
    """
    x0_list, lb_list, ub_list = [], [], []

    # from fitted peakks (A, x0, sigma, eta)
    for A, x0, sigma, eta in fitted_peaks:
        A = max(float(A), 0.0)
        sigma = max(float(sigma), 1e-3)
        eta = min(max(float(eta), 0.0), 1.0)

        x0_list += [A, x0, sigma, eta]
        lb_list += [
            0.0,
            x0 - pos_window,
            sigma / width_scale,
            0.0,
        ]
        ub_list += [
            3.0 * max(A, 1e-6),
            x0 + pos_window,
            sigma * width_scale,
            1.0,
        ]

    b = list(baseline) + [0.0, 0.0]
    x0_list += [float(b[0]), float(b[1])]
    lb_list += [-np.inf, -np.inf]
    ub_list += [np.inf, np.inf]

    return np.array(x0_list), (np.array(lb_list), np.array(ub_list))


# ----------------------------------------------------------------------
# Composite model: sum of peaks + linear baseline
# ----------------------------------------------------------------------
PARAMS_PER_PEAK = 4  # A, x0, sigma, eta  (pseudo-voigt)


# The optimizer only understands flat 1D arrays. 
# build_initial_guess flattens the peak parameters in to the following form
# params = [A1, x0_1, sigma1, eta1,  A2, x0_2, sigma2, eta2,  A3, x0_3, sigma3, eta3,  b0, b1]
def unpack(params, n_peaks):
    # reshaped the first n_peaks * PARAMS_PER_PEAK elements back into n_peaks-by-PARAMS_PER_PEAK
    peaks = params[: n_peaks * PARAMS_PER_PEAK].reshape(n_peaks, PARAMS_PER_PEAK)
    b0, b1 = params[n_peaks * PARAMS_PER_PEAK :] # the last two are b0 and b1
    return peaks, b0, b1


# Reconstruction through parameters
def model(params, x, n_peaks):
    peaks, b0, b1 = unpack(params, n_peaks)
    # set up baseline first
    y = b0 + b1 * x
    for A, x0, sigma, eta in peaks:
        y += pseudo_voigt(x, A, x0, sigma, eta)
    return y


def residuals(params, x, y, n_peaks):
    return model(params, x, n_peaks) - y


# ----------------------------------------------------------------------
# Main refinement routine
# ----------------------------------------------------------------------
def refine(freq, raw_signal, predicted_peaks, pos_window=5.0,
           width_scale=3.0, loss="soft_l1", f_scale=None, verbose=1,
           from_fitted=False, resume_baseline=(0.0, 0.0)):
    """`from_fitted=True` -> `predicted_peaks` is an already-refined
    [A, x0, sigma, eta] list and the fit resumes from it (see
    `build_guess_from_fitted`), carrying `resume_baseline` (b0, b1) forward;
    otherwise it's FCN output (A, pos, gamma)."""
    freq = np.asarray(freq, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)
    n_peaks = len(predicted_peaks)

    if from_fitted:
        x0, bounds = build_guess_from_fitted(predicted_peaks, pos_window,
                                             width_scale, baseline=resume_baseline)
    else:
        x0, bounds = build_initial_guess(predicted_peaks, pos_window, width_scale)

    if f_scale is None:
        # rough noise scale estimate; tune for the data
        f_scale = 0.05 * np.nanmax(np.abs(raw_signal))

    result = least_squares(
        residuals,
        x0,
        bounds=bounds,
        args=(freq, raw_signal, n_peaks),
        loss=loss,          # robust loss softens influence of tail mismatch
        f_scale=f_scale,
        method="trf",
        x_scale="jac",      # auto-scale params (A~1, x0~500, sigma~5, eta~0.5)
        verbose=verbose,
    )

    peaks_refined, b0, b1 = unpack(result.x, n_peaks)
    fitted_curve = model(result.x, freq, n_peaks)

    return {
        "success": result.success,
        "params": result.x,
        "peaks": peaks_refined,      # array of [A, x0, sigma, eta] per peak
        "baseline": (b0, b1),
        "fitted_curve": fitted_curve,
        "residual": raw_signal - fitted_curve,
        "cost": result.cost,
        "raw_result": result,
    }


# ----------------------------------------------------------------------
# Grouping / regional refinement for spectra with many peaks (25+)
#
# Idea: instead of handing the optimizer all N peaks + a single linear
# baseline at once (slow, and prone to cross-talk between distant
# peaks), split the peak list into smaller groups and fit each group's
# local window independently with `refine()`.
#
# IMPORTANT: a group boundary is only "free" if the peaks on either
# side of it are actually resolved. So there is no hard `max_peaks_per_group` cap. 
# `target_peaks_per_group` is a soft trigger: once a group reaches 
# that size, we can only cut at the next gap where the peaks are genuinely 
# separated -- position spacing >> gamma_i + gamma_j -- or at a real 
# low-signal ("baseline") stretch if one exists first. 
# A `max_group_multiple` safety valve prevents
# runaway growth in pathological cases where nothing is ever well
# separated -- past that size it cuts at the best (least-bad)
# available separation ratio found so far.
# ----------------------------------------------------------------------

def find_low_signal_splits(freq, raw_signal, threshold=0.1, min_run=5):
    """
    Locate candidate baseline regions: stretches where |raw_signal|
    stays at or below `threshold` for at least `min_run` consecutive
    samples. Returns the midpoint frequency of each such stretch,
    sorted ascending. These are frequencies where cutting the spectrum
    is "free" (no peak is being sliced through).

        threshold: signal level considered "nearly zero" (data units,
                   same scale as raw_signal -- tune to your noise floor)
        min_run:   minimum number of consecutive samples required for
                   a dip to count as a real baseline stretch rather
                   than noise or the narrow trough between two
                   overlapping peaks
    """
    freq = np.asarray(freq, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)
    below = np.abs(raw_signal) <= threshold

    splits = []
    run_start = None
    for i, flag in enumerate(below):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if i - run_start >= min_run:
                splits.append((freq[run_start] + freq[i - 1]) / 2.0)
            run_start = None
    if run_start is not None and len(freq) - run_start >= min_run:
        splits.append((freq[run_start] + freq[-1]) / 2.0)

    return np.array(sorted(splits))


def _well_separated(pos_i, gamma_i, pos_j, gamma_j, separation_factor):
    """True if two adjacent peaks' tails have decayed enough that a
    cut between them won't slice through meaningful shared signal."""
    denom = gamma_i + gamma_j
    if denom <= 0:
        return True
    return (pos_j - pos_i) > separation_factor * denom


def _proportional_boundary(pos_i, gamma_i, pos_j, gamma_j):
    """
    Place the cut between two peaks proportional to their widths
    rather than at the raw midpoint -- the wider peak gets more of the
    gap on its side, since its tail extends further.
    """
    total = gamma_i + gamma_j
    if total <= 0:
        return (pos_i + pos_j) / 2.0
    frac = gamma_i / total
    return pos_i + frac * (pos_j - pos_i)


def partition_peaks(freq, raw_signal, predicted_peaks, target_peaks_per_group=5,
                     separation_factor=3.0, low_signal_threshold=0.1, min_run=5,
                     max_group_multiple=10):
    """
    Split predicted_peaks (sorted by position) into groups in two passes:

      1. Cut at EVERY gap that's genuinely safe: either a real
         low-signal stretch, or a position gap that exceeds
         `separation_factor * (gamma_i + gamma_j)` (peaks actually
         resolved from one another). This defines natural "islands" of
         peaks and never merges two genuinely distinct islands
         together, regardless of how small either one is.
      2. Any island still bigger than `target_peaks_per_group *
         max_group_multiple` gets subdivided further, cutting at its
         internally-best (largest) separation ratio near a balanced
         split point -- there's no fully safe place to cut inside it,
         so this picks the least-bad option.

    `target_peaks_per_group` is therefore a soft target that mostly
    governs *how much oversized islands get subdivided*, not whether
    two distinct, well-separated peak clusters get cut apart -- that
    always happens regardless of resulting group size.

        separation_factor:   how many widths of "elbow room" between
                             two peaks counts as genuinely separated.
                             Larger = more conservative (fewer, bigger
                             groups); 3-5 is a reasonable starting range.
        max_group_multiple:  islands larger than
                             target_peaks_per_group * max_group_multiple
                             get force-subdivided at their best
                             available (least-bad) internal separation,
                             purely as a performance safety valve for
                             pathological spectra where a whole run of
                             peaks never resolves. 
                             WARNING: this is experimental, intentionally
                             a large number to avoid being activated

    Returns:
        groups:     list of peak-lists, e.g. [[(A,pos,gamma), ...], ...]
                    ordered left-to-right along freq
        boundaries: 1D array of length len(groups)+1, the freq edges
                    of each group's "core" region (boundaries[i] to
                    boundaries[i+1] belongs to groups[i]). Endpoints
                    are freq[0] and freq[-1].
    """
    freq = np.asarray(freq, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)

    # predicted_peaks are (A, pos, gamma): p[1] = pos, p[2] = gamma
    order = np.argsort([p[1] for p in predicted_peaks])
    sorted_peaks = [predicted_peaks[i] for i in order]
    positions = np.array([p[1] for p in sorted_peaks])
    gammas = np.array([max(p[2], 1e-3) for p in sorted_peaks])
    n = len(sorted_peaks)
    if n == 0:
        return [], np.array([freq[0], freq[-1]])

    baseline_splits = find_low_signal_splits(freq, raw_signal,
                                              low_signal_threshold, min_run)

    def gap_has_baseline(j):
        lo, hi = positions[j], positions[j + 1]
        return np.any((baseline_splits > lo) & (baseline_splits < hi))

    def gap_is_safe(j):
        return (_well_separated(positions[j], gammas[j],
                                positions[j + 1], gammas[j + 1],
                                separation_factor)
                )
                #or gap_has_baseline(j))

    # --- pass 1: natural islands at every genuinely safe gap ---
    island_bounds = [0]
    for j in range(n - 1):
        if gap_is_safe(j):
            island_bounds.append(j + 1)
    island_bounds.append(n)
    islands = [(island_bounds[k], island_bounds[k + 1])
               for k in range(len(island_bounds) - 1)]

    # --- pass 2: subdivide any island that's still oversized ---
    hard_cap = max(target_peaks_per_group * max_group_multiple,
                   target_peaks_per_group + 1)

    def subdivide(idx_lo, idx_hi):
        seg_n = idx_hi - idx_lo
        if seg_n <= hard_cap:
            return [(idx_lo, idx_hi)]
        seg_positions = positions[idx_lo:idx_hi]
        seg_gammas = gammas[idx_lo:idx_hi]
        denom = seg_gammas[:-1] + seg_gammas[1:]
        ratios = np.diff(seg_positions) / np.where(denom > 0, denom, 1e-9)
        ideal = seg_n // 2
        # prefer the widest available ratio, tie-broken toward balance
        best_local = min(range(len(ratios)), key=lambda k: (-ratios[k], abs(k - ideal)))
        cut = idx_lo + best_local + 1
        cut = min(max(cut, idx_lo + 1), idx_hi - 1)
        return subdivide(idx_lo, cut) + subdivide(cut, idx_hi)

    ranges = []
    for lo, hi in islands:
        ranges.extend(subdivide(lo, hi))
    ranges.sort(key=lambda r: r[0])

    groups = [[sorted_peaks[k] for k in range(lo, hi)] for lo, hi in ranges]

    boundaries = [freq[0]]
    for k in range(len(ranges) - 1):
        _, hi = ranges[k]
        lo2, _ = ranges[k + 1]
        pos_i, gamma_i = positions[hi - 1], gammas[hi - 1]
        pos_j, gamma_j = positions[lo2], gammas[lo2]
        prop = _proportional_boundary(pos_i, gamma_i, pos_j, gamma_j)

        # snap to the nearest true signal minimum within a small window
        # around the width-proportional point, when data supports it
        win = max(min(gamma_i, gamma_j), 1e-3)
        mask = (freq >= prop - win) & (freq <= prop + win)
        if np.any(mask):
            sub_freq = freq[mask]
            sub_sig = np.abs(raw_signal[mask])
            boundaries.append(sub_freq[np.argmin(sub_sig)])
        else:
            boundaries.append(prop)
    boundaries.append(freq[-1])

    return groups, np.array(boundaries)


# Experimetnal, not reliable
def estimate_global_baseline(freq, raw_signal, predicted_peaks, degree=1,
                              window_factor=3.0, n_iter=3, clip_sigma=3.0):
    """
    Fit ONE polynomial baseline (degree `degree`) for the whole
    spectrum, ONLY using the samples that fall outside
    `window_factor * gamma` of every predicted peak. A few rounds of
    sigma-clipping reject any leftover peak shoulders or outliers that
    slip into the baseline-only mask.

    This exists because per-group baselines (fit independently on each
    group's small window) are only weakly constrained and routinely
    disagree with their neighbors by a large margin -- that
    disagreement is what produces a patchy, non-uniform "background"
    when the group fits are stitched together. A single shared
    baseline, fit once (over the whole spectrum, away from every peak)
    and subtracted before any peak fitting happens, removes that
    failure mode entirely: every group is then just fitting peaks
    against an already-flat signal.

        degree: polynomial order. 1 = linear (b0 + b1*x), matching the
                rest of this module's baseline convention. Raise it if
                your background genuinely curves (e.g. broad Raman
                fluorescence humps) -- but since the fit only uses
                sparse, masked-out points and is unconstrained outside
                that support, keep it modest (2-3) unless you have a
                specific reason to go higher; too-high degree on
                sparse/clustered baseline points can ring or diverge
                near the spectrum edges (Runge's phenomenon).

    Returns a tuple of `degree + 1` coefficients, ascending order
    (c0, c1, c2, ...) such that baseline(x) = sum(c_k * x**k). For
    degree=1 this is exactly (b0, b1), matching `refine()`'s baseline
    format. Falls back to all-zero coefficients if no baseline-only
    samples are found (e.g. peaks cover the whole spectrum).
    """
    freq = np.asarray(freq, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)

    mask = np.ones_like(freq, dtype=bool)
    for A, pos, gamma in predicted_peaks:
        gamma = max(gamma, 1e-3)
        mask &= ~((freq >= pos - window_factor * gamma) &
                  (freq <= pos + window_factor * gamma))

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


def _eval_baseline(coeffs, x):
    """Evaluate a baseline given ascending-order coefficients (c0, c1, ...)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for k, c in enumerate(coeffs):
        y = y + c * x ** k
    return y


def _unpack_peaks_only(params, n_peaks):
    return params.reshape(n_peaks, PARAMS_PER_PEAK)


def _model_peaks_only(params, x, n_peaks):
    peaks = _unpack_peaks_only(params, n_peaks)
    y = np.zeros_like(x, dtype=float)
    for A, x0, sigma, eta in peaks:
        y = y + pseudo_voigt(x, A, x0, sigma, eta)
    return y


def _residuals_peaks_only(params, x, y, n_peaks):
    return _model_peaks_only(params, x, n_peaks) - y


def _refine_peaks_only(freq, target, predicted_peaks, pos_window=5.0,
                        width_scale=3.0, loss="soft_l1", f_scale=None, verbose=0,
                        from_fitted=False):
    """
    Same as `refine()`, but with NO baseline term at all -- used inside
    `refine_grouped` once a single global baseline has already been
    subtracted from `target`, so each group only ever fits peaks.

    `from_fitted=True` -> `predicted_peaks` is an already-refined
    [A, x0, sigma, eta] list; resume from it instead of FCN (A, pos, gamma).
    """
    freq = np.asarray(freq, dtype=float)
    target = np.asarray(target, dtype=float)
    n_peaks = len(predicted_peaks)

    if from_fitted:
        x0_full, (lb_full, ub_full) = build_guess_from_fitted(predicted_peaks, pos_window, width_scale)
    else:
        x0_full, (lb_full, ub_full) = build_initial_guess(predicted_peaks, pos_window, width_scale)
    x0, lb, ub = x0_full[:-2], lb_full[:-2], ub_full[:-2]  # drop b0, b1 slots

    if f_scale is None:
        peak_amp = np.nanmax(np.abs(target)) if target.size else 1.0
        f_scale = 0.05 * peak_amp if peak_amp > 0 else 1.0

    result = least_squares(
        _residuals_peaks_only, x0, bounds=(lb, ub),
        args=(freq, target, n_peaks),
        loss=loss, f_scale=f_scale, method="trf", x_scale="jac",
        verbose=verbose,
    )

    peaks_refined = _unpack_peaks_only(result.x, n_peaks)
    fitted_curve = _model_peaks_only(result.x, freq, n_peaks)

    return {
        "success": result.success,
        "params": result.x,
        "peaks": peaks_refined,
        "fitted_curve": fitted_curve,
        "residual": target - fitted_curve,
        "cost": result.cost,
        "raw_result": result,
    }



#
#  Following splits the peaks into group and optimize separately
#  Not as good as I like
#

def refine_grouped(freq, raw_signal, predicted_peaks, target_peaks_per_group=5,
                   separation_factor=3.0, low_signal_threshold=0.1, min_run=5,
                   max_group_multiple=20, pad=15.0, pos_window=5.0, width_scale=3.0,
                   fit_baseline=False, baseline_degree=1, baseline_window_factor=3.0,
                   loss="soft_l1", f_scale=None, verbose=0, from_fitted=False):
    """
    Replacement for `refine()` when you have many peaks (25+):
    fits ONE global baseline for the whole spectrum, subtracts it,
    then partitions predicted_peaks into groups (see `partition_peaks`)
    and fits each group's peaks independently over a locally padded
    window of the baseline-subtracted signal. Results are stitched
    back into one full-length curve with the shared baseline added
    back.

    Groups never fit their own baseline -- that's the point. A shared
    baseline is what keeps the reconstructed spectrum's background
    continuous across group boundaries instead of patchy.

        pad: extra frequency margin (in x-units) added on each side of
             a group's core region when *fitting*, so peaks near a
             boundary still have enough neighborhood to constrain their
             tails. Only the core region (not the padding) is written
             into the stitched output, so overlapping fits from
             neighboring groups never double-count.
        baseline_degree: polynomial order for the shared global
             baseline (1 = linear, matching `refine()`). See
             `estimate_global_baseline` for guidance on going higher.
        baseline_window_factor: how many peak-widths (gamma) around
             each predicted peak to exclude when estimating the global
             baseline (see `estimate_global_baseline`).
        from_fitted: if True, `predicted_peaks` is an already-refined
             [A, x0, sigma, eta] list and each group resumes from those
             values (see `build_guess_from_fitted`) instead of restarting
             from FCN (A, pos, gamma). Re-fitting an unedited list then
             converges immediately.

    Returns a dict:
        peaks:         array of [A, x0, sigma, eta] stacked across all
                       groups, in left-to-right (sorted position) order
        baseline:      tuple of `baseline_degree + 1` coefficients,
                       ascending order (c0, c1, ...) such that
                       baseline(x) = sum(c_k * x**k). For
                       baseline_degree=1 this is exactly (b0, b1),
                       the same shape as `refine()`'s output, so
                       existing plotting code that does
                       `b0, b1 = fit_result["baseline"]` keeps working
                       unchanged. For baseline_degree>1 that unpacking
                       needs updating -- e.g.
                       `baseline = sum(c*freq**k for k,c in enumerate(fit_result["baseline"]))`.
        fitted_curve:  full-length stitched model curve (baseline + peaks)
        residual:      raw_signal - fitted_curve
        boundaries:    group boundary frequencies (see partition_peaks)
        groups:        the partitioned predicted_peaks, per group
        group_results: list of per-group dicts (peaks-only fit output:
                       params, peaks, success, cost, raw_result, ...)
                       plus its (core_lo, core_hi) boundary -- useful
                       for inspecting/debugging individual group fits
    """
    freq = np.asarray(freq, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)

    # use with caution, still experimenting
    if fit_baseline:
        baseline_coeffs = estimate_global_baseline(
            freq, raw_signal, predicted_peaks,
            degree=baseline_degree, window_factor=baseline_window_factor)
        baseline_vals = _eval_baseline(baseline_coeffs, freq)
        target_signal = raw_signal - baseline_vals  # flat signal, groups only see this
    else:
        baseline_coeffs = (0.0,0.0)
        target_signal = raw_signal

    # partition using the flattened signal so "nearly zero" is judged
    # relative to the actual background, not a possibly-sloped raw one.
    # from_fitted peaks carry sigma (~2*gamma) in slot [2] instead of the
    # FCN gamma partition_peaks expects, so halve separation_factor to keep
    # the island cuts at the same physical scale as the initial fit.
    part_sep = separation_factor / 2.0 if from_fitted else separation_factor
    groups, boundaries = partition_peaks(
        freq, target_signal, predicted_peaks,
        target_peaks_per_group=target_peaks_per_group,
        separation_factor=part_sep,
        low_signal_threshold=low_signal_threshold,
        min_run=min_run,
        max_group_multiple=max_group_multiple,
    )

    fitted_peaks_only = np.zeros_like(raw_signal)
    all_peaks = []
    group_results = []

    for g_idx, peaks_in_group in enumerate(groups):
        core_lo, core_hi = boundaries[g_idx], boundaries[g_idx + 1]

        # Cap this group's padding so its fit window can never reach far
        # enough to pull in a neighboring peak's real signal. A fixed
        # `pad` sized for widely-spaced, broad peaks can massively
        # over-extend in a densely-spaced, narrow-peak region -- if the
        # padded window includes a neighbor's real signal that this
        # group's own peaks aren't modeling, the fit has no way to
        # explain that extra signal except by distorting its own
        # amplitude/width to compensate.
        #
        # Cap against the nearest neighboring PEAK's actual position
        # (not the neighboring group's whole core width, which can
        # still extend far past a nearby valley if that neighbor's own
        # core stretches out further on its far side).
        local_pad = pad
        if g_idx > 0:
            nearest_left_peak = groups[g_idx - 1][-1][1]
            local_pad = min(local_pad, 0.5 * max(core_lo - nearest_left_peak, 0.0))
        if g_idx < len(groups) - 1:
            nearest_right_peak = groups[g_idx + 1][0][1]
            local_pad = min(local_pad, 0.5 * max(nearest_right_peak - core_hi, 0.0))
        local_pad = max(local_pad, 0.0)

        fit_mask = (freq >= core_lo - local_pad) & (freq <= core_hi + local_pad)
        if not np.any(fit_mask):
            continue

        sub_freq = freq[fit_mask]
        sub_target = target_signal[fit_mask]

        result = _refine_peaks_only(
            sub_freq, sub_target, peaks_in_group,
            pos_window=pos_window, width_scale=width_scale,
            loss=loss, f_scale=f_scale, verbose=verbose,
            from_fitted=from_fitted,
        )

        # accumulate this group's fitted peaks over the FULL freq range
        # (not just its core region) and sum across groups. Each
        # predicted peak belongs to exactly one group, so there's no
        # double-counting risk -- but clipping the curve at the core
        # boundary was wrong: a broad/isolated peak's fitted width can
        # legitimately need more room to decay than its own core region
        # provides, and hard-clipping it there produced a visible
        # truncation cliff *and* left real signal unexplained just past
        # the boundary, which then distorted the neighboring group's
        # fit trying to compensate for it.
        fitted_peaks_only += _model_peaks_only(
            result["params"], freq, len(peaks_in_group))

        all_peaks.append(result["peaks"])
        group_results.append({
            "core_bounds": (core_lo, core_hi),
            "n_peaks": len(peaks_in_group),
            **result,
        })

    combined_peaks = (np.vstack(all_peaks) if all_peaks 
                      else np.empty((0, PARAMS_PER_PEAK)))
    fitted_curve = fitted_peaks_only
    if fit_baseline:
        fitted_curve += baseline_vals
    residual = raw_signal - fitted_curve

    return {
        "peaks": combined_peaks,
        "baseline": baseline_coeffs,
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
#    print("Baseline (b0, b1):", out["baseline"])
#    print("Residual RMS:", np.sqrt(np.mean(out["residual"] ** 2)))
#
#    # --- for 25+ peaks, use the grouped refinement instead ---
#    # out = refine_grouped(
#    #     freq, raw_signal, predicted_peaks,
#    #     target_peaks_per_group=5,    # soft target group size
#    #     separation_factor=3.0,       # peaks must be this many widths apart to cut between them
#    #     low_signal_threshold=0.1,    # "nearly zero" cutoff, tune to your noise floor
#    #     min_run=5,                   # min consecutive samples to count as baseline
#    #     pad=15.0,                    # extra x-units of context around each group when fitting
#    # )
#    # print("n groups:", len(out["groups"]))
#    # print("Refined peaks (A, x0, sigma, eta):")
#    # for row in out["peaks"]:
#    #     print(row)
#    # print("Residual RMS:", np.sqrt(np.mean(out["residual"] ** 2)))
