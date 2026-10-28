"""
Uncertainties: bootstrap refits, credible intervals from posterior samples,
and Monte Carlo error propagation.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .fitting import Fitter, LeastSquares


def _refit_many(wave, error, model_obj, fitter, p0, resampled_fluxes):
    """Refit every resampled flux array; failed fits are skipped."""
    samples = []
    for flux_resampled in resampled_fluxes:
        try:
            samples.append(fitter.fit(wave, flux_resampled, error, model_obj, p0=p0).params)
        except Exception:
            continue

    samples = np.array(samples)
    if samples.size == 0:
        raise RuntimeError("All bootstrap iterations failed to converge.")
    return np.median(samples, axis=0), np.std(samples, axis=0), samples


def bootstrap_errors(wave, flux, error, model_obj, n_boot: int = 200,
                     fitter: Optional[Fitter] = None, seed: Optional[int] = None):
    """
    Parametric bootstrap: perturb the flux with Gaussian noise of the given
    1-sigma errors and refit ``n_boot`` times.

    Parameters
    ----------
    wave, flux, error : ndarray
        If ``error`` is None, 5 per cent of the flux standard deviation is used.
    model_obj : object
        Anything with ``model``, ``initial_guess`` and ``bounds``.
    n_boot : int
    fitter : Fitter, optional
        Defaults to :class:`agnspec.fitting.LeastSquares`.
    seed : int, optional

    Returns
    -------
    param_median, param_std : ndarray
    samples : ndarray, shape (n_ok, n_params)
        Best-fit parameters of every successful refit.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float) if error is not None else np.full_like(flux, np.std(flux) * 0.05)

    fitter = fitter or LeastSquares()
    rng = np.random.default_rng(seed)
    p0 = model_obj.initial_guess(wave, flux)

    fluxes = (flux + rng.normal(scale=error) for _ in range(n_boot))
    return _refit_many(wave, error, model_obj, fitter, p0, fluxes)


def residual_bootstrap_errors(wave, flux, error, model_obj, best_fit_params,
                              n_boot: int = 200, fitter: Optional[Fitter] = None,
                              seed: Optional[int] = None):
    """
    Residual bootstrap: resample the residuals of the best fit with
    replacement and refit. Useful when the noise is not well described by the
    quoted errors.

    Parameters
    ----------
    best_fit_params : sequence
        Parameter vector from an earlier fit; also the starting point of every refit.

    The other parameters and the return values are as in :func:`bootstrap_errors`.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    fitter = fitter or LeastSquares()
    rng = np.random.default_rng(seed)

    model_flux = model_obj.model(wave, *best_fit_params)
    residuals = flux - model_flux

    fluxes = (model_flux + rng.choice(residuals, size=residuals.size, replace=True)
              for _ in range(n_boot))
    return _refit_many(wave, error, model_obj, fitter, best_fit_params, fluxes)


def credible_interval(samples, level: float = 0.68):
    """
    Equal-tailed credible interval from posterior samples (for example from
    :class:`agnspec.fitting.MCMC`).

    Parameters
    ----------
    samples : ndarray, shape (n_samples,) or (n_samples, n_params)
    level : float
        Probability mass inside the interval (0.68 is about 1 sigma).

    Returns
    -------
    median, lower, upper : ndarray or float
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
    Monte Carlo error propagation through ``func``, for example FWHM and
    L5100 uncertainties through :func:`agnspec.measurements.virial_blackhole_mass`.

    Parameters
    ----------
    func : callable
        ``func(*values) -> float``
    values : sequence of float
        Central value of each input.
    errors : sequence of float
        1-sigma error of each input (independent and Gaussian).
    n_samples : int
    seed : int, optional

    Returns
    -------
    median, sigma : float
    samples : ndarray
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    draws = rng.normal(loc=values, scale=errors, size=(n_samples, len(values)))
    samples = np.array([func(*row) for row in draws])

    return float(np.median(samples)), float(np.std(samples)), samples
