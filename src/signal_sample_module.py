import numpy as np
from .lineshapes import lorentzian_np, voigt, gaussian

rng = np.random.default_rng(0)

 
def sample_amplitude_skewed(rng, small_range=(0.3,2), large_range=(2,12), large_prob=0.10):
    """Mostly small peaks, occasionally a large one -- directly controls
    the proportion via large_prob instead of relying on uniform sampling
    over the full range to incidentally produce enough small examples."""
    if rng.random() < large_prob:
        return rng.uniform(*large_range)
    else:
        return rng.uniform(*small_range)


def sample_true_peaks(rng, cfg, min_sep_factor=1.0, min_sep_offset=1.0, amp_kwargs=None):
    """Generate a list of tuples (A, pos, gamma). Each tuple defines a peak.

    Separation requirement between any two peaks is now gamma-aware:
    |pos_i - pos_j| > min_sep_factor*(gamma_i + gamma_j) + min_sep_offset
    instead of a fixed min_sep, since resolvability scales with linewidth
    (empirically, separation > gamma + ~1.0 is roughly the detectable regime).

    amp_kwargs: optional dict forwarded to sample_amplitude_skewed to override
    the default small/large amplitude split -- used by the "skewed" pattern
    generator to push the distribution further toward small peaks.
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

            A = sample_amplitude_skewed(rng, **(amp_kwargs or {}))
            peaks.append((A, pos, gamma))
    return peaks


def generate_sandwiched_cluster(rng, amp_range, gamma_range, pos_range,
                                window_width_range=(30,80), small_peak_frac_range=(0.015,0.2),
                                fac=1.0, max_tries=300, min_sep_offset=1.0):
    """fac: minimum-separation hyperparameter -- two peaks must be at
    least fac*(gamma_a+gamma_b) + min_sep_offset apart, so genuinely wide 
    peaks can't be placed on top of each other even in a narrow window. 
    Positions and gammas are placed together via rejection sampling (same pattern as
    the original sample_true_peaks); amplitudes are assigned afterward,
    once final position order is known, so the flanking-large/middle-
    small pattern still applies correctly regardless of placement order."""
    window_width = rng.uniform(*window_width_range)
    n_peaks_target = rng.integers(3,8)
    center = rng.uniform(pos_range[1]*0.15, pos_range[1]*0.85)

    placed_pg = []  # (pos, gamma) 2-tuples -- amplitude assigned later, by position order
    tries = 0
    while len(placed_pg) < n_peaks_target and tries < max_tries:
        tries += 1
        pos = center + rng.uniform(-window_width/2, window_width/2)
        gamma = rng.uniform(*gamma_range)
        if all(abs(pos - p[0]) >= fac*(gamma + p[1]) + min_sep_offset for p in placed_pg):
            placed_pg.append((pos, gamma))

    placed_pg.sort(key=lambda p: p[0])  # sort by position -- first/last become the flanking peaks
    n_placed = len(placed_pg)
    peaks = []
    for i, (pos, gamma) in enumerate(placed_pg):
        # first and last (taller)
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
    """Convert a list of (A, pos, gamma) peaks to a summed peak signal."""
    signal = np.zeros_like(W)
    for A, pos, gamma in peaks:
        if peak_type=='lorentzian':
            signal += lorentzian_np(W, A, pos, gamma)
        elif peak_type=='voigt':
            signal += voigt(W, A, pos, gamma, sigma=0.2)
        else:
            signal += gaussian(W, A, pos, gamma, sigma=gamma)

    return signal.astype(np.float32)


# --------------------------------------------------------------------------- #
# Peak-set generators
#
# Each takes (n, rng, cfg) and returns a list of length n, where every element
# is itself a list of (A, pos, gamma) tuples -- one spectrum's worth of peaks.
# They produce *peaks only*; rendering to a signal (lineshapes + Poisson noise
# + optional feature channels) is done once, centrally, by
# render_peaks_to_dataset. This keeps each pattern independently tunable and
# testable, and lets new patterns be added without touching the render path.
# --------------------------------------------------------------------------- #
def gen_random(n, rng, cfg):
    """Fully random peaks with the default (mild) small-amplitude skew."""
    return [sample_true_peaks(rng, cfg) for _ in range(n)]


def gen_sandwiched(n, rng, cfg):
    """The known-hard pattern: small peaks sandwiched between large neighbors
    in a narrow window, plus 0..(n_peak_max-2) fully random filler peaks."""
    out = []
    for _ in range(n):
        peaks = generate_sandwiched_cluster(rng,
                                            cfg.amp_range,
                                            cfg.gamma_range,
                                            cfg.pos_range,
                                            window_width_range=cfg.window_width_range)
        n_extra = rng.integers(0, cfg.n_peak_max - 2)  # middle slots only
        for _ in range(n_extra):
            peaks.append((rng.uniform(*cfg.amp_range),
                          rng.uniform(*cfg.pos_range),
                          rng.uniform(*cfg.gamma_range)))
        out.append(peaks)
    return out


def gen_skewed(n, rng, cfg):
    """Random layout but with the amplitude distribution pushed hard toward
    small peaks (cfg.skew_* knobs), the other empirically-hard regime."""
    amp_kwargs = dict(small_range=cfg.skew_small_range,
                      large_range=cfg.skew_large_range,
                      large_prob=cfg.skew_large_prob)
    return [sample_true_peaks(rng, cfg, amp_kwargs=amp_kwargs) for _ in range(n)]


# Registry -- extension point: a new pattern = one gen_* function + one entry
# here + one key in cfg.dataset_mix. Nothing else changes.
PATTERN_GENERATORS = {
    "random": gen_random,
    "sandwiched": gen_sandwiched,
    "skewed": gen_skewed,
}


def render_peaks_to_dataset(peaks_list, cfg, add_noise=True):
    """Render a list of peak-lists into the model input array X.

    Owns the whole signal path: lineshape synthesis -> Poisson measurement ->
    optional derivative feature channels. Returns X with shape
    (n, n_points) for single-channel, or (n, N_INPUT_CHANNELS, n_points).
    """
    X = []
    for peaks in peaks_list:
        raw = convert_peaks_to_signal(peaks, cfg.W, peak_type='lorentzian')
        if add_noise:
            raw = poisson_measurement(raw, scale=cfg.scale, dark=cfg.dark, seed=None)

        if cfg.N_INPUT_CHANNELS > 1:
            from src.features import average_features
            X.append(average_features(raw, window=11, polyorder=3, deriv=0))
        else:
            X.append(raw)
    return np.array(X)


def _allocate_counts(n_samples, mix):
    """Split n_samples across mix keys by (normalized) fraction, using
    largest-remainder rounding so the counts sum to exactly n_samples."""
    unknown = set(mix) - set(PATTERN_GENERATORS)
    if unknown:
        raise KeyError(f"unknown dataset_mix pattern(s): {sorted(unknown)}; "
                       f"known: {sorted(PATTERN_GENERATORS)}")

    keys = list(mix)
    weights = np.array([mix[k] for k in keys], dtype=float)
    if weights.sum() <= 0:
        raise ValueError(f"dataset_mix weights must be positive, got {mix}")
    raw = weights / weights.sum() * n_samples
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[:n_samples - counts.sum()]:
        counts[i] += 1
    return dict(zip(keys, counts.tolist()))


def build_mixed_dataset(n_samples, rng, cfg, mix=None, add_noise=True):
    """Generate a dataset as a concatenation of per-pattern blocks.

    mix: ordered {pattern_key: fraction}; defaults to cfg.dataset_mix. Fractions
    are normalized, so they need not sum to exactly 1.0. Blocks are shuffled
    together before rendering so patterns aren't left contiguous.

    Returns (X, peaks_list) -- the same contract as the old generate_dataset.
    """
    if mix is None:
        mix = cfg.dataset_mix

    counts = _allocate_counts(n_samples, mix)

    peaks_list = []
    for key, count in counts.items():
        if count:
            peaks_list.extend(PATTERN_GENERATORS[key](count, rng, cfg))

    perm = rng.permutation(len(peaks_list))
    peaks_list = [peaks_list[i] for i in perm]

    X = render_peaks_to_dataset(peaks_list, cfg, add_noise=add_noise)
    return X, peaks_list


def generate_dataset(n_samples, rng, cfg, add_noise=True):
    """Backward-compatible entry point -- builds a dataset from cfg.dataset_mix."""
    return build_mixed_dataset(n_samples, rng, cfg, add_noise=add_noise)


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

    peaks = []
    for pos in positions:
        A = rng.uniform(*amp_range)
        gamma = gamma_fixed if gamma_fixed is not None else rng.uniform(*gamma_range)
        peaks.append((A, pos, gamma))
    return peaks
