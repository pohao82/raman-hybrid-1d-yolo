import numpy as np

def filter_by_raw_signal_flat(peaks, W, raw_signal, cfg, threshold):
    """For each predicted peak, check whether the RAW signal itself
    actually reaches threshold within that peak's grid cell -- a
    prediction with no supporting raw signal there is almost certainly
    the model firing on noise/context rather than a real feature."""
    kept = []
    for conf, A, pos, gamma in peaks:
        g = int(np.argmin(np.abs(cfg.w_grid - pos)))
        cell_start = cfg.w_grid[g]
        cell_end = cell_start + cfg.grid_spacing
        mask = (W >= cell_start) & (W < cell_end)
        if mask.sum() == 0:
            continue
        local_max = raw_signal[mask].max()
        if local_max >= threshold:
            kept.append((conf, A, pos, gamma))
    return kept


#
# Below, the filter is based on moving average
#


from scipy.signal import savgol_filter


#
# Multiscale moving average difference
#
def compute_multiscale_diff_movavg(signal, candidates, window_bank, W,
                                   span=15, polyorder=1, scale=200.0):
    """
    Compute a multi-scale i_diff profile for each candidate peak.Baseline

    For every adjacent pair of window sizes in `window_bank`, this computes
    the dual-window Savitzky-Golay i_diff metric
        i_diff = sqrt( (smooth_short - smooth_long)^2 * scale )
    smoothed once more (window=7) to reduce single-sample spikes, then takes
    the local max within +/-span samples of each candidate's position.

    Each candidate therefore gets one i_diff value per scale-pair, letting
    downstream code accept/reject based on whichever scale best matches
    that peak's width (see idiff_accept-style logic).

    Parameters
    ----------
    signal : 1D array
        Intensity values on the `W` grid.
    candidates : list of (id, A, pos, gamma)
        FCN-detected candidates; `pos` is in the same physical units as `W`
        (e.g. cm^-1), NOT an array index.
    window_bank : list of int
        Sorted Savitzky-Golay window sizes, e.g. [5, 9, 15, 25, 41, 71].
        Consecutive entries form the scale-pairs used for i_diff.
    W : 1D array
        Physical-unit grid matching `signal` (used to convert `pos` -> index).
    span : int
        Half-width (in samples) of the local window used to pull out the
        i_diff value around each candidate.
    polyorder : int
        Polynomial order for all Savitzky-Golay smooths (short/long window
        smooths and the final i_diff smoothing pass).
    scale : float
        Multiplicative scale factor inside the i_diff formula.

    Returns
    -------
    list of np.ndarray
        One array per candidate, length == len(window_bank) - 1, holding
        the local-max i_diff value at each adjacent scale-pair.
    """
    # Precompute each window's smooth once for the whole spectrum (reused
    # across all candidates and all scale-pairs that touch this window).
    smooths = {
        w: savgol_filter(signal, window_length=w, polyorder=polyorder)
        for w in window_bank
    }

    profiles = []
    for _, A, pos, gamma in candidates:
        # pos is in physical units (e.g. cm^-1) -- map to a grid index via W,
        # not int(round(pos)), which would silently misindex the signal.
        idx = np.clip(np.searchsorted(W, pos), 0, len(signal) - 1)
        lo, hi = max(0, idx - span), min(len(signal), idx + span + 1)

        vals = []
        for w_short, w_long in zip(window_bank[:-1], window_bank[1:]):
            i_diff = (smooths[w_short] - smooths[w_long]) ** 2 * scale
            i_diff = savgol_filter(i_diff ** 0.5, window_length=7, polyorder=polyorder)
            vals.append(i_diff[lo:hi].max())

        profiles.append(np.array(vals))

    return profiles


#
# Multiscale gaussian difference DoG
#

from scipy.ndimage import gaussian_filter1d


def compute_multiscale_diff_gaussian(signal, candidates, window_bank, W,
                                     span=15, polyorder=1, scale=200.0):
    """
    candidates: list of (_, A, pos, gamma) tuples, or raw position values —
    position lookup against W happens internally.
    """
    signal = np.asarray(signal, dtype=float)
    W = np.asarray(W, dtype=float)

    # accept either full peak tuples or bare positions
    positions = np.array([c[2] if hasattr(c, "__len__") else c for c in candidates])
    cand_idx = np.array([np.argmin(np.abs(W - p)) for p in positions], dtype=int)

    if span % 2 == 0:
        span += 1
    baseline = savgol_filter(signal, window_length=span, polyorder=polyorder, mode='interp')
    residual = signal - baseline

    sigmas = np.asarray(window_bank, dtype=float) / 6.0
    smoothed = [gaussian_filter1d(residual, sigma=s, mode='nearest') for s in sigmas]

    n_channels = len(sigmas) - 1
    idiff = np.zeros((len(cand_idx), n_channels))
    for i in range(n_channels):
        diff = smoothed[i] - smoothed[i + 1]
        idiff[:, i] = (diff[cand_idx] ** 2) * scale
    return idiff


# -----------------------------------------
#     testing for now (not used yet)
# -----------------------------------------

def _valid_savgol_window(window_length, n_samples, polyorder):
    """
    Coerce a requested Savitzky-Golay window into something scipy will
    accept: odd, >= polyorder + 1, and <= n_samples (also forced odd).
    Keeps the hyperparameter usable as-is (e.g. 10, 30) without the
    caller having to worry about scipy's parity/length constraints.
    """
    w = int(window_length)
    if w % 2 == 0:
        w += 1

    max_w = n_samples if n_samples % 2 == 1 else n_samples - 1
    w = min(w, max_w)

    min_w = polyorder + 1
    if min_w % 2 == 0:
        min_w += 1

    return max(w, min_w)


def filter_by_raw_signal(peaks, W, raw_signal, cfg, threshold=0.01,
                          window_short=10, window_long=30, polyorder=1,
                          scale_factor=100.0):
    """
    For each predicted peak, check whether the RAW signal shows real
    structure at that peak's grid cell -- a prediction with no
    supporting structure there is almost certainly the model firing on
    noise/context rather than a real feature.

    "Real structure" is judged by comparing a short-window and a
    long-window Savitzky-Golay smooth of the raw signal:

        i_diff = (smooth_short - smooth_long)**2 * scale_factor

    In a flat noise-only region the two smooths track each other
    closely (both are just averaging out noise), so i_diff stays near
    zero. Near a real peak the two smooths disagree -- the short
    window follows the peak's actual shape while the long window
    averages over it -- so i_diff spikes right where real signal is.
    This only needs to be checked in the immediate neighborhood of
    each predicted peak, not across the whole spectrum: we're not
    trying to find peaks, just confirm the ones already predicted.

    window_short, window_long, polyorder, scale_factor, and threshold
    are all hyperparameters -- start here and retune once more data is
    available. window_short/window_long should be chosen relative to
    the narrowest real peak width (in grid points) you expect: too
    close together and genuine peaks won't produce enough of a
    mismatch to clear the threshold; too far apart and the long smooth
    starts washing out real peak shape everywhere, inflating i_diff
    even in flat regions.

    Returns the same shape as the original function: a list of
    (A, pos, gamma) for peaks that pass.
    """
    W = np.asarray(W, dtype=float)
    raw_signal = np.asarray(raw_signal, dtype=float)
    n = len(raw_signal)

    w_short = _valid_savgol_window(window_short, n, polyorder)
    w_long = _valid_savgol_window(window_long, n, polyorder)

    smooth_short = savgol_filter(raw_signal, window_length=w_short, polyorder=polyorder)
    smooth_long = savgol_filter(raw_signal, window_length=w_long, polyorder=polyorder)
    i_diff = (smooth_short - smooth_long) ** 2 * scale_factor

    kept = []
    for _, A, pos, gamma in peaks:
        g = int(np.argmin(np.abs(cfg.w_grid - pos)))
        cell_start = cfg.w_grid[g]
        cell_end = cell_start + cfg.grid_spacing
        mask = (W >= cell_start) & (W < cell_end)
        if mask.sum() == 0:
            continue
        local_max = i_diff[mask].max() # small window, comparable to or smaller than your gamma range
        if local_max >= threshold:
            kept.append((A, pos, gamma))
    return kept


def compute_i_diff(raw_signal, window_short=10, window_long=30,
                    polyorder=1, scale_factor=100.0):
    """Full-spectrum i_diff = (smooth_short - smooth_long)**2 * scale_factor."""
    raw_signal = np.asarray(raw_signal, dtype=float)
    n = len(raw_signal)
    w_short = _valid_savgol_window(window_short, n, polyorder)
    w_long = _valid_savgol_window(window_long, n, polyorder)
    s_short = savgol_filter(raw_signal, window_length=w_short, polyorder=polyorder)
    s_long = savgol_filter(raw_signal, window_length=w_long, polyorder=polyorder)
    return (s_short - s_long) ** 2 * scale_factor


def estimate_i_diff_threshold(raw_signal, window_short=10, window_long=30,
                               polyorder=1, scale_factor=100.0, percentile=95.0):
    """
    Estimate a per-spectrum i_diff threshold as a percentile of its own
    i_diff distribution. Recompute this once per spectrum at load time
    -- not a fixed constant across datasets.
    """
    i_diff = compute_i_diff(raw_signal, window_short, window_long,
                             polyorder, scale_factor)
    return float(np.percentile(i_diff, percentile))


def analyze_i_diff(raw_signal, window_short=10, window_long=30,
                    polyorder=1, scale_factor=100.0, percentile=95.0):
    """
    Quick diagnostic: distribution summary of i_diff + resulting
    threshold + fraction of spectrum that would pass it. Use this to
    sanity check the percentile choice on each new spectrum before
    trusting it.
    """
    i_diff = compute_i_diff(raw_signal, window_short, window_long,
                             polyorder, scale_factor)
    threshold = float(np.percentile(i_diff, percentile))
    frac_above = float(np.mean(i_diff >= threshold))

    return {
        "threshold": threshold,
        "median": float(np.median(i_diff)),
        "mean": float(np.mean(i_diff)),
        "max": float(np.max(i_diff)),
        "p90": float(np.percentile(i_diff, 90)),
        "p95": float(np.percentile(i_diff, 95)),
        "p99": float(np.percentile(i_diff, 99)),
        "frac_above_threshold": frac_above,  # sanity check: should roughly match (100-percentile)/100
    }

