"""
Spectrum preprocessing: rest-frame shift, Galactic dereddening, bad-pixel
masking and smoothing for display.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def to_rest_frame(wave, z):
    """
    Shift the wavelength axis to the rest frame, lambda_rest = lambda_obs / (1 + z).

    Only the wavelength changes. Flux density and errors are left exactly as
    they are, so integrals over the rest-frame axis are smaller than the
    observed-frame ones by a factor of (1 + z).

    Parameters
    ----------
    wave : ndarray
        Observed-frame wavelength (Angstrom).
    z : float
        Redshift.

    Returns
    -------
    wave_rest : ndarray
    """
    return np.asarray(wave, dtype=float) / (1.0 + z)


def _ccm89_alambda(wave, a_v, rv=3.1):
    """
    Cardelli, Clayton & Mathis (1989) extinction curve, A_lambda in mag.
    Pure-numpy version used when the ``extinction`` package is not installed.
    Valid from 1000 A to 3.3 micron (x = 0.3 - 10 per micron).
    """
    wave = np.asarray(wave, dtype=float)
    x = 1e4 / wave                       # inverse microns
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    ir = (x >= 0.3) & (x < 1.1)
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61

    op = (x >= 1.1) & (x < 3.3)
    y = x[op] - 1.82
    a[op] = (1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3
             + 0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6
             + 0.32999 * y**7)
    b[op] = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3
             - 5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6
             - 2.09002 * y**7)

    uv = (x >= 3.3) & (x < 8.0)
    xu = x[uv]
    fa = np.where(xu > 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0)
    fb = np.where(xu > 5.9, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.0)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb

    fuv = (x >= 8.0) & (x <= 10.0)
    xf = x[fuv] - 8.0
    a[fuv] = -1.073 - 0.628 * xf + 0.137 * xf**2 - 0.070 * xf**3
    b[fuv] = 13.670 + 4.257 * xf - 0.420 * xf**2 + 0.374 * xf**3

    if np.any((x < 0.3) | (x > 10.0)):
        raise ValueError("CCM89 is only defined for 1000 A - 3.3 um.")
    return a_v * (a + b / rv)


def deredden_spectrum(wave, flux, error=None, ebv=None, av=None, rv=3.1, law="ccm89"):
    """
    Correct a spectrum for Galactic (foreground) dust extinction.

    Give either ``ebv`` or ``av`` (not both); ``av = rv * ebv``. The
    correction must be applied at *observed* wavelengths.

    Parameters
    ----------
    wave, flux : ndarray
    error : ndarray, optional
        If given it is dereddened too and both arrays are returned.
    ebv, av : float
        E(B-V) or A_V in magnitudes.
    rv : float
        Total-to-selective extinction ratio.
    law : {"ccm89", "fitzpatrick99", "odonnell94", "calzetti00"}
        Extinction curve. Everything except ``ccm89`` needs the optional
        ``extinction`` package.

    Returns
    -------
    flux_dered or (flux_dered, error_dered)
    """
    try:
        import extinction
    except ImportError:
        extinction = None

    if (ebv is None) == (av is None):
        raise ValueError("Provide exactly one of `ebv` or `av`.")

    a_v = av if av is not None else rv * ebv

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if a_v == 0:
        return flux.copy() if error is None else (flux.copy(), np.asarray(error, dtype=float).copy())

    if extinction is None:
        if law.lower() != "ccm89":
            raise ImportError(
                f"law='{law}' needs the `extinction` package "
                "(pip install extinction); only 'ccm89' works without it.")
        a_lambda = _ccm89_alambda(wave, a_v, rv)
    else:
        law_map = {
            "ccm89": extinction.ccm89,
            "fitzpatrick99": extinction.fitzpatrick99,
            "odonnell94": extinction.odonnell94,
            "calzetti00": extinction.calzetti00,
        }
        if law.lower() not in law_map:
            raise ValueError(f"Unknown extinction law '{law}'. Known: {list(law_map)}")
        a_lambda = law_map[law.lower()](wave, a_v, rv)

    flux_dered = flux * 10 ** (0.4 * a_lambda)
    if error is None:
        return flux_dered

    error_dered = np.asarray(error, dtype=float) * 10 ** (0.4 * a_lambda)
    return flux_dered, error_dered


def mask_bad_pixels(wave, flux, error=None, bad_regions=None,
                    sigma_clip=None, mask_nan=True):
    """
    Build a mask that drops NaNs, zero or negative errors, telluric bands and
    (optionally) cosmic-ray-like outliers.

    Parameters
    ----------
    wave, flux : ndarray
    error : ndarray, optional
    bad_regions : list of (float, float), optional
        Wavelength intervals to drop, e.g. ``[(6860, 6890), (7590, 7650)]``.
    sigma_clip : float, optional
        If given, drop pixels that sit more than this many sigma away from a
        smoothed copy of the spectrum.
    mask_nan : bool
        Drop NaN/Inf flux or error values and non-positive errors.

    Returns
    -------
    good_mask : ndarray of bool
        True for pixels to keep.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.ones_like(wave, dtype=bool)

    if mask_nan:
        good &= np.isfinite(wave) & np.isfinite(flux)
        if error is not None:
            good &= np.isfinite(error) & (np.asarray(error) > 0)

    if bad_regions:
        for lo, hi in bad_regions:
            good &= ~((wave >= lo) & (wave <= hi))

    if sigma_clip is not None:
        smooth = gaussian_filter1d(np.nan_to_num(flux), sigma=5)
        resid = flux - smooth
        std = np.nanstd(resid[good])
        good &= np.abs(resid) < sigma_clip * std

    return good


def smooth_spectrum(flux, sigma=3):
    """
    Gaussian-smooth a spectrum, for plotting only (do not fit the result).

    Parameters
    ----------
    flux : ndarray
    sigma : float
        Kernel width in pixels.
    """
    return gaussian_filter1d(np.nan_to_num(np.asarray(flux, dtype=float)), sigma=sigma)
