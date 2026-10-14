"""
uncertainties.py
=================

Uncertainty estimation for fitted parameters and derived
measurements: flux-error bootstrap resampling, posterior-based
credible intervals (from MCMC/nested-sampling chains), and simple
first-order error propagation for derived quantities like FWHM,
luminosity, and black-hole mass.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .fitting import Fitter, LeastSquares, FitResult


def bootstrap_errors(wave, flux, error, model_obj, n_boot: int = 200,
                      fitter: Optional[Fitter] = None, seed: Optional[int] = None):
    """
    Estimate parameter uncertainties by resampling the flux within
    its errors (parametric bootstrap) and refitting.

    Parameters
    ----------
    wave, flux, error : ndarray
    model_obj : object
        Model exposing ``model``, ``initial_guess``, ``bounds``.
    n_boot : int
        Number of bootstrap iterations.
    fitter : Fitter, optional
        Defaults to :class:`fitting.LeastSquares`.
    seed : int, optional

    Returns
    -------
    param_median : ndarray
    param_std : ndarray
    samples : ndarray, shape (n_boot, n_params)
        All bootstrap best-fit parameter vectors (failed iterations
        are dropped).
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float) if error is not None else np.full_like(flux, np.std(flux) * 0.05)

    fitter = fitter or LeastSquares()
    rng = np.random.default_rng(seed)

    samples = []
    p0 = model_obj.initial_guess(wave, flux)
    for _ in range(n_boot):
        flux_resampled = flux + rng.normal(scale=error)
        try:
            result = fitter.fit(wave, flux_resampled, error, model_obj, p0=p0)
            samples.append(result.params)
        except Exception:
            continue

    samples = np.array(samples)
    if samples.size == 0:
        raise RuntimeError("All bootstrap iterations failed to converge.")

    return np.median(samples, axis=0), np.std(samples, axis=0), samples


def residual_bootstrap_errors(wave, flux, error, model_obj, best_fit_params,
                               n_boot: int = 200, fitter: Optional[Fitter] = None,
                               seed: Optional[int] = None):
    """
    Bootstrap by resampling (with replacement) the residuals of the
    best fit, rather than drawing fresh Gaussian noise -- useful when
    the true noise model is uncertain or non-Gaussian.

    Parameters
    ----------
    wave, flux, error : ndarray
    model_obj : object
    best_fit_params : sequence
        Best-fit parameter vector from an initial fit.
    n_boot : int
    fitter : Fitter, optional
    seed : int, optional

    Returns
    -------
    param_median, param_std, samples : as in :func:`bootstrap_errors`
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    fitter = fitter or LeastSquares()
    rng = np.random.default_rng(seed)

    model_flux = model_obj.model(wave, *best_fit_params)
    residuals = flux - model_flux

    samples = []
    for _ in range(n_boot):
        boot_resid = rng.choice(residuals, size=residuals.size, replace=True)
        flux_resampled = model_flux + boot_resid
        try:
            result = fitter.fit(wave, flux_resampled, error, model_obj, p0=best_fit_params)
            samples.append(result.params)
        except Exception:
            continue

    samples = np.array(samples)
    if samples.size == 0:
        raise RuntimeError("All bootstrap iterations failed to converge.")

    return np.median(samples, axis=0), np.std(samples, axis=0), samples


def credible_interval(samples, level: float = 0.68):
    """
    Compute a symmetric (median +/- half-width) credible interval
    from posterior samples (e.g. from :class:`fitting.MCMC` or
    :class:`fitting.NestedSampling`).

    Parameters
    ----------
    samples : ndarray, shape (n_samples,) or (n_samples, n_params)
    level : float
        Credible interval mass (e.g. 0.68 for ~1 sigma, 0.95 for ~2
        sigma).

    Returns
    -------
    median, lower, upper : ndarray or float
        Median and the lower/upper bounds of the credible interval.
    """
    samples = np.asarray(samples)
    lo_pct = 100 * (1 - level) / 2
    hi_pct = 100 * (1 + level) / 2
    median = np.median(samples, axis=0)
    lower = np.percentile(samples, lo_pct, axis=0)
    upper = np.percentile(samples, hi_pct, axis=0)
    return median, lower, upper


def propagate_errors(func: Callable, values: Sequence[float], errors: Sequence[float],
                      n_samples: int = 10000, seed: Optional[int] = None):
    """
    Monte Carlo error propagation through an arbitrary function of
    several correlated-or-not parameters, e.g. propagating FWHM and
    L5100 errors through :func:`measurements.virial_blackhole_mass`.

    Parameters
    ----------
    func : callable
        ``func(*values) -> float``
    values : sequence of float
        Best-fit central values of each input.
    errors : sequence of float
        1-sigma uncertainties of each input (assumed independent,
        Gaussian).
    n_samples : int
    seed : int, optional

    Returns
    -------
    median, sigma : float
        Monte Carlo estimate of the propagated central value and
        uncertainty.
    samples : ndarray
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    draws = rng.normal(loc=values, scale=errors, size=(n_samples, len(values)))
    samples = np.array([func(*row) for row in draws])

    return float(np.median(samples)), float(np.std(samples)), samples
