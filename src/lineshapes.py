import numpy as np
from scipy.special import voigt_profile

# ----------------------------------------------------------------------
#   Lineshapes functions
# ----------------------------------------------------------------------

def lorentzian_np(w, A, pos, gamma):
    return A * gamma**2 / ((w - pos)**2 + gamma**2)


def voigt(W, A, pos, gamma, sigma=0.2):
    """Voigt profile scaled to height A.
    Parameters:
    - W: Grid/array of x-values (e.g., wavelength/frequency)
    - A: Peak height (amplitude)
    - pos: Center position of the peak
    - gamma: Lorentzian half-width at half-maximum (HWHM)
    - sigma: Gaussian standard deviation
    """
    # Shift input grid to center around 0
    x = W - pos
    # Calculate unscaled profile (area = 1)
    profile = voigt_profile(x, sigma, gamma)
    # Calculate peak value at x = 0 for height scaling
    peak_val = voigt_profile(0, sigma, gamma)
    # Scale so the maximum height equals A
    return A * (profile / peak_val)


def pseudo_voigt(x, A, x0, sigma, eta):
    """eta=1 -> pure Lorentzian, eta=0 -> pure Gaussian. sigma ~ FWHM."""
    sigma = max(sigma, 1e-8)
    L = A / (1.0 + (2.0 * (x - x0) / sigma) ** 2)
    G = A * np.exp(-4.0 * np.log(2.0) * ((x - x0) / sigma) ** 2)
    return eta * L + (1.0 - eta) * G


def gaussian(W, A, pos, gamma, sigma):
    """Gaussian function matching the signature: (W, A, pos, gamma)

    Parameters:
    - W: Array/grid of values (e.g., wavelength, energy, position)
    - A: Height scale (peak height)
    - pos: Position of the center (peak center)
    - gamma: Width parameter (sigma / standard deviation)
    """
    return A * np.exp(-(((W - pos) / sigma) ** 2) / 2)

