"""
utils.py
========

Shared constants, unit conversions, and small numerical helpers used
throughout the ``agnspec`` package.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Physical constants (cgs unless noted)
# ----------------------------------------------------------------------
C_KMS = 2.99792458e5          # speed of light, km/s
C_ANGS = 2.99792458e18        # speed of light, Angstrom/s
H_PLANCK = 6.62607015e-27     # erg s
K_BOLTZ = 1.380649e-16        # erg / K

# ----------------------------------------------------------------------
# Common AGN rest-frame emission-line wavelengths (vacuum, Angstrom)
# ----------------------------------------------------------------------
LINE_WAVELENGTHS = {
    "Lyalpha": 1215.67,
    "CIV": 1549.06,
    "CIII]": 1908.73,
    "MgII": 2798.75,
    "OII": 3727.09,
    "Hdelta": 4101.73,
    "Hgamma": 4340.47,
    "Hbeta": 4861.33,
    "OIII_4959": 4958.91,
    "OIII_5007": 5006.84,
    "HeI_5876": 5875.62,
    "OI_6300": 6300.30,
    "Halpha": 6562.80,
    "NII_6548": 6548.05,
    "NII_6584": 6583.45,
    "SII_6716": 6716.44,
    "SII_6731": 6730.82,
}


def gaussian(x, amplitude, center, sigma):
    """
    Simple Gaussian profile.

    Parameters
    ----------
    x : ndarray
        Wavelength array (Angstrom).
    amplitude : float
        Peak amplitude (flux density units).
    center : float
        Central wavelength.
    sigma : float
        Gaussian sigma (Angstrom).

    Returns
    -------
    ndarray
    """
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def multi_gaussian(x, *params):
    """
    Sum of N Gaussians. ``params`` must be a flat sequence of
    ``(amplitude, center, sigma)`` triplets.
    """
    x = np.asarray(x, dtype=float)
    model = np.zeros_like(x)
    n = len(params) // 3
    for i in range(n):
        amp, cen, sig = params[3 * i: 3 * i + 3]
        model += gaussian(x, amp, cen, sig)
    return model


def air_to_vac(wave_air):
    """
    Convert air wavelengths (Angstrom, valid above ~2000 A) to vacuum
    using the Morton (2000) / IAU standard formula.
    """
    wave_air = np.asarray(wave_air, dtype=float)
    s = 1e4 / wave_air
    n = (1 + 0.00008336624212083 +
         0.02408926869968 / (130.1065924522 - s ** 2) +
         0.0001599740894897 / (38.92568793293 - s ** 2))
    return wave_air * n


def vac_to_air(wave_vac):
    """
    Convert vacuum wavelengths (Angstrom) to air wavelengths using the
    inverse of the IAU standard conversion (Morton 2000).
    """
    wave_vac = np.asarray(wave_vac, dtype=float)
    s = 1e4 / wave_vac
    n = (1 + 0.0000834254 +
         0.02406147 / (130 - s ** 2) +
         0.00015998 / (38.9 - s ** 2))
    return wave_vac / n


def fwhm_to_sigma(fwhm):
    """Convert a Gaussian FWHM to sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_to_fwhm(sigma):
    """Convert a Gaussian sigma to FWHM."""
    return sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))


def wave_to_velocity(wave, wave0):
    """
    Convert a wavelength array to velocity space (km/s) relative to a
    rest wavelength, using the non-relativistic Doppler formula.
    """
    wave = np.asarray(wave, dtype=float)
    return C_KMS * (wave - wave0) / wave0


def velocity_to_wave(vel, wave0):
    """Inverse of :func:`wave_to_velocity`."""
    vel = np.asarray(vel, dtype=float)
    return wave0 * (1.0 + vel / C_KMS)


def nearest_index(array, value):
    """Return the index of ``array`` closest to ``value``."""
    array = np.asarray(array)
    return int(np.argmin(np.abs(array - value)))


def flux_density_to_luminosity(flux, distance_cm):
    """
    Convert an observed flux (erg/s/cm^2 or erg/s/cm^2/A) to luminosity
    (erg/s or erg/s/A) given a luminosity distance in cm, assuming
    isotropic emission: L = 4 pi d_L^2 F.
    """
    return 4.0 * np.pi * distance_cm ** 2 * flux


def ensure_array(x):
    """Coerce input into a 1-D numpy float array."""
    return np.atleast_1d(np.asarray(x, dtype=float))
