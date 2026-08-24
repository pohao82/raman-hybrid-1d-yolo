import numpy as np
from .lineshapes import lorentzian_np, voigt, gaussian

rng = np.random.default_rng(0)

 
def sample_amplitude_skewed(rng, small_range=(0.3,5), large_range=(5,16), large_prob=0.15):
    """Mostly small peaks, occasionally a large one -- directly controls
    the proportion via large_prob instead of relying on uniform sampling
    over the full range to incidentally produce enough small examples."""
    if rng.random() < large_prob:
        return rng.uniform(*large_range)
    else:
        return rng.uniform(*small_range)


def sample_true_peaks(rng, cfg, min_sep_factor=1.0, min_sep_offset=1.0):
    """Generate a list of tuples (A, pos, gamma). Each tuple defines a peak.

    Separation requirement between any two peaks is now gamma-aware:
    |pos_i - pos_j| > min_sep_factor*(gamma_i + gamma_j) + min_sep_offset
    instead of a fixed min_sep, since resolvability scales with linewidth
    (empirically, separation > gamma + ~1.0 is roughly the detectable regime).
    """
    gamma_range = cfg.gamma_range
    pos_range = cfg.pos_range
    K = cfg.K  # maximum number of peaks

    n = rng.integers(1, K + 1)
    peaks = []  # (A, pos, gamma)
    tries = 0
    while len(peaks) < n and tries < 200:
        tries += 1
        pos = rng.uniform(*pos_range)
        gamma = rng.uniform(*gamma_range)

        if all(
            abs(pos - p[1]) > min_sep_factor * (gamma + p[2]) + min_sep_offset
            for p in peaks
        ):

            A = sample_amplitude_skewed(rng)
            peaks.append((A, pos, gamma))
    return peaks


def generate_sandwiched_cluster(rng, amp_range, gamma_range, pos_range,
                                window_width_range=(30,70), small_peak_frac_range=(0.015,0.2),
                                fac=1.0, max_tries=300, min_sep_offset=1.0):
    """fac: minimum-separation hyperparameter -- two peaks must be at
    least fac*(gamma_a+gamma_b) + min_sep_offset apart, so genuinely wide 
    peaks can't be placed on top of each other even in a narrow window. 
    Positions and gammas are placed together via rejection sampling (same pattern as
    the original sample_true_peaks); amplitudes are assigned afterward,
    once final position order is known, so the flanking-large/middle-
    small pattern still applies correctly regardless of placement order."""
    window_width = rng.uniform(*window_width_range)
    n_peaks_target = rng.integers(3,6)
    center = rng.uniform(pos_range[1]*0.15, pos_range[1]*0.85)

    placed = []  # (pos, gamma) only
    tries = 0
    while len(placed) < n_peaks_target and tries < max_tries:
        tries += 1
        pos = center + rng.uniform(-window_width/2, window_width/2)
        gamma = rng.uniform(*gamma_range)
        if all(abs(pos - p[0]) >= fac*(gamma + p[1]) + min_sep_offset for p in placed):
            placed.append((pos, gamma))

    placed.sort(key=lambda p: p[0])  # sort by position -- first/last become the flanking peaks
    n_placed = len(placed)
    peaks = []
    for i, (pos, gamma) in enumerate(placed):
        if i==0 or i==n_placed-1:
            A = rng.uniform(0.25,1.0)*(amp_range[1]-amp_range[0]) + amp_range[0]
        else:
            frac = rng.uniform(*small_peak_frac_range)
            A = frac*(amp_range[1]-amp_range[0]) + amp_range[0]
        peaks.append((A, pos, gamma))
    return peaks


def poisson_measurement(clean, scale=2000.0, dark=0.02, seed=None):
    # Photon-counting noise: Poisson-distributed counts, converted back to
    # intensity units. This is the physically correct noise model for Raman
    # correct instead) -- the noise model choice follows the detectoru
    # physics, not a default.
    local_rng = np.random.default_rng(seed)
    lam = np.clip(clean + dark, 0, None) * scale
    counts = local_rng.poisson(lam)
    return counts / scale


def convert_peaks_to_signal(peaks, W, peak_type='lorentzian'):
    """Convert peaks parameters (A, Gamma, POS) to peak signal"""
    signal = np.zeros_like(W)
    for A, pos, gamma in peaks:
        if peak_type=='lorentzian':
            signal += lorentzian_np(W, A, pos, gamma)
        elif peak_type=='voigt':
            signal += voigt(W, A, pos, gamma, sigma=0.2)
        else:
            signal += gaussian(W, A, pos, gamma, sigma=gamma)

    return signal.astype(np.float32)


def generate_dataset(n_samples, rng, cfg, add_noise=True):

    # special patterns
    window_width_range = cfg.window_width_range
    sandwiched_prob = cfg.sandwiched_prob
    # maxium number of peaks in a small cluster
    n_peak_max = cfg.n_peak_max # need to be >= 3

    # general peak parameters
    amp_range = cfg.amp_range
    gamma_range = cfg.gamma_range
    pos_range = cfg.pos_range
    # noise
    scale = cfg.scale
    dark = cfg.dark
    seed = None

    X, peaks_list = [], []
    for _ in range(n_samples):
        # additional conditionsl for different patterns
        if rng.random() < sandwiched_prob:
            # Deliberately inject the known-hard pattern (small peaks sandwiched 
            # between large neighbors)
            peaks = generate_sandwiched_cluster(rng, 
                                                amp_range, 
                                                gamma_range, 
                                                pos_range, 
                                                window_width_range=window_width_range)
            n_extra = rng.integers(0, n_peak_max-2) # remove the max and min, n_peak-2 in the middle
            for _ in range(n_extra):
                peaks.append((rng.uniform(*amp_range),
                              rng.uniform(*pos_range),
                              rng.uniform(*gamma_range)))
        else:
            # fully random
            peaks = sample_true_peaks(rng, cfg)

        raw = convert_peaks_to_signal(peaks, cfg.W, peak_type='lorentzian')
        if add_noise:
            raw = poisson_measurement(raw, scale=scale, dark=dark, seed=seed)

        if cfg.N_INPUT_CHANNELS > 1:
            from src.features import average_features
            avg_feats = average_features(raw, window=11, polyorder=3, deriv=0)
            X.append(avg_feats)
        else:
            X.append(raw)

        peaks_list.append(peaks)
    return np.array(X), peaks_list


#
# Additional testing kit for stress test, easier to specify parameters mannually than passing cfg
# 
def generate_close_peaks_sample(rng, separations, amp_range, gamma_range,
                                center=300.0, gamma_fixed=None):
    """Generate one test sample with peaks placed at a specific, controlled
    spacing -- deliberately stressing the min_sep/grid-resolution assumption,
    unlike the training generator which enforces a minimum separation.

    separations: list of gaps (cm^-1) between consecutive peaks, e.g.
        [15] -> 2 peaks 15 cm^-1 apart
        [10, 10] -> 3 peaks, each 10 cm^-1 from its neighbor
    """
    n = len(separations) + 1
    positions = [center]
    for sep in separations:
        positions.append(positions[-1] + sep)

    true_peaks = []
    for pos in positions:
        A = rng.uniform(*amp_range)
        gamma = gamma_fixed if gamma_fixed is not None else rng.uniform(*gamma_range)
        true_peaks.append((A, pos, gamma))
    return true_peaks
