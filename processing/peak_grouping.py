"""
Peak grouping for regional (divide-and-conquer) refinement.

Given the peaks and the spectrum, decide where the list can be cut into
smaller, independently-fittable "islands", and where each island's core
region ends. `processing.refinement_fit.refine_grouped` consumes the
`(groups, boundaries)` result; it is the only caller.

# ----------------------------------------------------------------------
# Grouping / regional refinement for spectra with many peaks (25+)
#
# Idea: instead of handing the optimizer all N peaks at once (slow, and
# prone to cross-talk between distant peaks), split the peak list into
# smaller groups and fit each group's local window independently.
# `refine()` is the degenerate single-group case of this.
#
# IMPORTANT: a group boundary is only "free" if the peaks on either
# side of it are actually resolved.
# `target_peaks_per_group` is a soft trigger: once a group reaches
# that size, only cut at the next gap where the peaks are genuinely
# separated -- position spacing >> gamma_i + gamma_j.
# A `max_group_multiple` safety valve prevents
# runaway growth in pathological cases where nothing is ever well
# separated -- past that size it cuts at the best (least-bad)
# available separation ratio found so far.
# ----------------------------------------------------------------------
"""

import numpy as np


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
                     separation_factor=3.0, max_group_multiple=10,
                     single_group=False):
    """
    Split predicted_peaks (sorted by position) into groups in two passes:

      1. Cut at EVERY gap that's genuinely safe: a position gap that
         exceeds `separation_factor * (gamma_i + gamma_j)` (peaks
         actually resolved from one another). This defines natural
         "islands" of peaks and never merges two genuinely distinct
         islands together, regardless of how small either one is.
      2. Any island still bigger than `target_peaks_per_group *
         max_group_multiple` gets subdivided further, cutting at its
         internally-best (largest) separation ratio near a balanced
         split point -- there's no fully safe place to cut inside it,
         so this picks the least-bad option.

    `single_group=True` short-circuits both passes: every peak goes into
    one group spanning the whole spectrum. This is how `refine()` (the
    global simultaneous fit) is expressed as the degenerate case of
    `refine_grouped()`.

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

    if single_group:
        return [sorted_peaks], np.array([freq[0], freq[-1]])

    def gap_is_safe(j):
        return _well_separated(positions[j], gammas[j],
                               positions[j + 1], gammas[j + 1],
                               separation_factor)

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
