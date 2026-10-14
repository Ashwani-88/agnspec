"""
preprocessing.py
================

Spectrum preprocessing: rest-frame transformation, Galactic-extinction
correction, bad-pixel masking, and smoothing for visualization.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


def to_rest_frame(wave, z):
    """
    Convert observed spectrum to rest frame. Only shift wavelength to rest frame.

    lambda_rest = lambda_obs / (1 + z)
    

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
    wave = np.asarray(wave, dtype=float)
    wave_rest = wave / (1.0 + z)

    return wave_rest


def deredden_spectrum(wave, flux,error=None, ebv=None, av=None, rv=3.1, law="ccm89" ):
    """
    Correct a spectrum for Galactic (foreground) dust extinction.

    Provide *either* `ebv` or `av` (not both) -- `av = rv * ebv`.
    """
    import extinction

    if (ebv is None) == (av is None):
        raise ValueError("Provide exactly one of `ebv` or `av`.")

    a_v = av if av is not None else rv * ebv

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if a_v == 0:
        return flux.copy() if error is None else (flux.copy(), np.asarray(error, dtype=float).copy())

    law_map = {
        "ccm89": extinction.ccm89,
        "fitzpatrick99": extinction.fitzpatrick99,
        "odonnell94": extinction.odonnell94,
        "calzetti00": extinction.calzetti00,
    }
    ext_func = law_map.get(law.lower(), extinction.ccm89)
    a_lambda = ext_func(wave, a_v, rv)

    flux_dered = flux * 10 ** (0.4 * a_lambda)
    if error is None:
        return flux_dered

    error_dered = np.asarray(error, dtype=float) * 10 ** (0.4 * a_lambda)
    return flux_dered, error_dered

def mask_bad_pixels(wave, flux, error=None, bad_regions=None,
                     sigma_clip=None, mask_nan=True):
    """
    Mask telluric regions, cosmic rays, NaNs, and other bad pixels.

    Parameters
    ----------
    wave : ndarray
    flux : ndarray
    error : ndarray, optional
    bad_regions : list of (float, float), optional
        Wavelength intervals to mask out (e.g. telluric bands,
        chip gaps): ``[(6860, 6890), (7590, 7650)]``.
    sigma_clip : float, optional
        If given, flag flux pixels deviating by more than this many
        sigma from a smoothed version of the spectrum as cosmic rays.
    mask_nan : bool
        Mask NaN/Inf flux or error values.

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
    Gaussian smoothing for visualization (does not alter the data
    used for fitting; use only for plotting/inspection).

    Parameters
    ----------
    flux : ndarray
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    ndarray
        Smoothed flux array, same shape as input.
    """
    flux = np.asarray(flux, dtype=float)
    return gaussian_filter1d(np.nan_to_num(flux), sigma=sigma)
    
    
def trim_spectrum(wavelength, flux, flux_err=None, wmin=None, wmax=None):
    """
    Trim a spectrum to a wavelength range [wmin, wmax].
    """
    wavelength = np.asarray(wavelength, dtype=float)
    mask = (wavelength >= (wmin if wmin is not None else -np.inf)) & \
           (wavelength <= (wmax if wmax is not None else np.inf))

    if flux_err is None:
        return wavelength[mask], np.asarray(flux)[mask]
    return wavelength[mask], np.asarray(flux)[mask], np.asarray(flux_err)[mask]
