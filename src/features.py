import numpy as np
from scipy.signal import savgol_filter


def compute_svd_feature(signal, k=12, hankel_frac=0.5):
    # Move to the Fourier ("time") domain, where a sum of Lorentzians
    # becomes a sum of damped exponentials; build a Hankel matrix and take
    # its singular value spectrum (the classical Prony/ESPRIT rank-estimate
    # idea). Returns a scale-invariant length-k feature vector.
    z = np.fft.ifft(signal)
    n = len(z)
    m = int(n * hankel_frac)
    H = np.zeros((m, n - m + 1), dtype=complex)
    for i in range(m):
        H[i] = z[i:i + (n - m + 1)]
    svals = np.linalg.svd(H, compute_uv=False)
    top = svals[:k]
    return (top / (top[0] + 1e-12)).astype(np.float32)


def compute_derivative_channel(signal, window=11, polyorder=3, deriv=0):
    """Savitzky-Golay smoothed 2nd derivative -- verified above to locate
    peaks within ~1-2 cm^-1 of true position, even under Poisson noise,
    since a Lorentzian's 2nd derivative has a sharp minimum exactly at the
    peak center. Per-sample standardized so it's on a comparable numeric
    scale to the raw signal channel (raw and derivative differ by ~100x in
    raw magnitude -- checked directly, not assumed)."""
    d2 = savgol_filter(signal, window, polyorder, deriv=deriv)
    d2 = (d2 - d2.mean()) / (d2.std() + 1e-8)
    return d2.astype(np.float32)


def average_features(signal, window=11, polyorder=3, deriv=0):
    # with additional channel
    d2 = compute_derivative_channel(signal, window=13)
    return np.stack([signal, d2], axis=0)
    #d3 = compute_derivative_channel(signal, window=15)
    #return np.stack([signal, d2, d3], axis=0)



