"""
measurements.py
================

Derived measurements from fitted models: equivalent widths, line
fluxes, kinematic widths, luminosities, and black-hole mass /
Eddington ratio estimates from single-epoch virial relations.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import simpson

from .utils import C_KMS, sigma_to_fwhm, fwhm_to_sigma


def equivalent_width(wave, flux, continuum):
    """
    Equivalent width of a spectral feature relative to a local
    continuum:

        EW = integral[ (continuum - flux) / continuum ] dlambda

    A positive EW corresponds to an emission line (flux above the
    continuum) with this sign convention flipped from the classic
    absorption-line definition; adjust the sign if you need the
    opposite convention.

    Parameters
    ----------
    wave : ndarray
    flux : ndarray
        Total observed (line + continuum) flux density.
    continuum : ndarray
        Continuum model flux density evaluated at the same ``wave``.

    Returns
    -------
    ew : float
        Equivalent width, same units as ``wave`` (typically Angstrom).
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    continuum = np.asarray(continuum, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = (flux - continuum) / continuum
    integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
    ew = simpson(integrand, x=wave)
    return float(ew)


def line_flux(model, wave=None):
    """
    Integrated flux of an emission-line model.

    Parameters
    ----------
    model : dict or ndarray or callable
        * If a dict with keys ``amplitude`` and ``sigma`` (as returned
          by :func:`emission_lines.fit_gaussian`), the analytic
          Gaussian integral ``sqrt(2 pi) * amplitude * sigma`` is
          used.
        * If an ndarray (a model flux array) and ``wave`` is given,
          numerically integrates via Simpson's rule.
    wave : ndarray, optional
        Required if ``model`` is an ndarray of flux values.

    Returns
    -------
    flux_int : float
        Integrated line flux (erg/s/cm^2 if the input flux density
        was in erg/s/cm^2/Angstrom).
    """
    if isinstance(model, dict) and "amplitude" in model and "sigma" in model:
        return float(np.sqrt(2.0 * np.pi) * model["amplitude"] * model["sigma"])

    if wave is None:
        raise ValueError("`wave` must be supplied when `model` is a flux array.")
    model = np.asarray(model, dtype=float)
    wave = np.asarray(wave, dtype=float)
    return float(simpson(model, x=wave))


def fwhm(gaussian_params):
    """
    Full width at half maximum of a Gaussian line profile.

    Parameters
    ----------
    gaussian_params : dict or float
        Either a dict containing ``"sigma"`` (Angstrom), or the sigma
        value itself.

    Returns
    -------
    fwhm_value : float
        FWHM in the same units as sigma (Angstrom).
    """
    sigma = gaussian_params["sigma"] if isinstance(gaussian_params, dict) else gaussian_params
    return float(sigma_to_fwhm(sigma))


def sigma_velocity(gaussian_params, rest_wave=None):
    """
    Convert a Gaussian line-width sigma (Angstrom) into a velocity
    dispersion (km/s).

    Parameters
    ----------
    gaussian_params : dict or float
        Dict with "sigma" and (if `rest_wave` not given) "center"
        keys, or the raw sigma value in Angstrom.
    rest_wave : float, optional
        Rest-frame central wavelength; if not provided, taken from
        ``gaussian_params["center"]``.

    Returns
    -------
    sigma_v : float
        Velocity dispersion in km/s.
    """
    if isinstance(gaussian_params, dict):
        sigma = gaussian_params["sigma"]
        wave0 = rest_wave if rest_wave is not None else gaussian_params["center"]
    else:
        sigma = gaussian_params
        if rest_wave is None:
            raise ValueError("rest_wave must be given if gaussian_params is a scalar sigma.")
        wave0 = rest_wave

    return float(C_KMS * sigma / wave0)


def velocity_shift(center, rest_wave):
    """
    Doppler velocity shift of a fitted line center relative to its
    rest-frame wavelength.

    Parameters
    ----------
    center : float
        Fitted (rest-frame, after redshift correction) line center.
    rest_wave : float
        Laboratory rest wavelength.

    Returns
    -------
    v : float
        Velocity shift in km/s (positive = redshifted / receding).
    """
    return float(C_KMS * (center - rest_wave) / rest_wave)


def line_luminosity(flux, luminosity_distance):
    """
    Convert an integrated line flux to a luminosity.

    Parameters
    ----------
    flux : float
        Integrated line flux (erg/s/cm^2).
    luminosity_distance : float
        Luminosity distance in cm.

    Returns
    -------
    L : float
        Luminosity in erg/s.
    """
    return float(4.0 * np.pi * luminosity_distance ** 2 * flux)


def virial_blackhole_mass(fwhm, l5100, calibration="vp06"):
    """
    Single-epoch virial black-hole mass estimate from the broad-line
    FWHM and the 5100 A continuum luminosity, following the standard
    form:

        log(M_BH / Msun) = a + b * log(L5100 / 1e44) + 2 * log(FWHM / 1000 km/s)

    Parameters
    ----------
    fwhm : float
        Broad-line FWHM in km/s (e.g. Hbeta).
    l5100 : float
        Monochromatic luminosity lambda*L_lambda at 5100 A, erg/s.
    calibration : str
        Coefficients to use:
          * "vp06"  : Vestergaard & Peterson (2006) Hbeta, a=6.91, b=0.50
          * "gh05"  : Greene & Ho (2005) Hbeta, a=6.67, b=0.63 (with FWHM
                       exponent 2 as well; note different luminosity
                       tracer conventions exist in the literature)

    Returns
    -------
    mbh : float
        Black-hole mass in solar masses.
    """
    coeffs = {
        "vp06": (6.91, 0.50),
        "gh05": (6.67, 0.63),
    }
    a, b = coeffs.get(calibration, coeffs["vp06"])

    log_l = np.log10(l5100 / 1e44)
    log_fwhm_term = 2.0 * np.log10(fwhm / 1000.0)
    log_mbh = a + b * log_l + log_fwhm_term
    return float(10 ** log_mbh)


def eddington_ratio(mbh, lbol):
    """
    Eddington ratio L_bol / L_Edd.

    Parameters
    ----------
    mbh : float
        Black-hole mass in solar masses.
    lbol : float
        Bolometric luminosity in erg/s.

    Returns
    -------
    lambda_edd : float
        Dimensionless Eddington ratio.
    """
    l_edd = 1.26e38 * mbh  # erg/s, for M in Msun
    return float(lbol / l_edd)


def bolometric_luminosity(l5100, bc=9.26):
    """
    Estimate AGN bolometric luminosity from the 5100 A monochromatic
    luminosity using a bolometric correction factor (Richards et al.
    2006 give BC ~ 9.26 for lambda*L_lambda(5100A)).

    Parameters
    ----------
    l5100 : float
        lambda*L_lambda at 5100 A, erg/s.
    bc : float
        Bolometric correction factor.

    Returns
    -------
    lbol : float
    """
    return float(bc * l5100)
