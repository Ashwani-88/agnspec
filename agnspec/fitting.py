"""
fitting.py
==========

Generic fitting engines ("Fitters") that operate on any object
exposing the small interface used throughout ``agnspec``:

    model(wave, *params)      -> ndarray
    initial_guess(wave, flux) -> sequence
    bounds()                  -> (lower, upper)
    param_names (or free_param_names) -> list of str

This includes :class:`continuum.ContinuumModel`,
:class:`continuum.PowerLaw`, :class:`emission_lines.LineComplex`, and
subclasses thereof, so the same ``LeastSquares``/``MCMC``/
``NestedSampling`` classes work for continuum and line fitting alike.

    Fitters
    ├── LeastSquares
    ├── MCMC
    └── NestedSampling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


def _get_param_names(model_obj):
    return getattr(model_obj, "free_param_names", None) or getattr(model_obj, "param_names", None) or []


@dataclass
class FitResult:
    """Container for the outcome of any Fitter.fit() call."""

    params: np.ndarray
    param_names: Sequence[str]
    param_errors: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None      # posterior samples (MCMC/Nested)
    log_evidence: Optional[float] = None       # nested sampling only
    chi2: Optional[float] = None
    dof: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

    def as_dict(self):
        d = dict(zip(self.param_names, self.params))
        if self.param_errors is not None:
            for name, err in zip(self.param_names, self.param_errors):
                d[f"{name}_err"] = err
        return d


class Fitter:
    """Base class for all fitting backends."""

    def fit(self, wave, flux, error, model_obj, p0=None) -> FitResult:
        raise NotImplementedError


# ----------------------------------------------------------------------
# Least squares (fast, default)
# ----------------------------------------------------------------------
class LeastSquares(Fitter):
    """
    Weighted nonlinear least-squares fitting via
    ``scipy.optimize.curve_fit`` (Levenberg-Marquardt / trust-region
    reflective, depending on whether bounds are supplied).
    """

    def __init__(self, maxfev: int = 50000):
        self.maxfev = maxfev

    def fit(self, wave, flux, error, model_obj, p0=None) -> FitResult:
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        sigma = np.asarray(error, dtype=float) if error is not None else None

        if p0 is None:
            p0 = model_obj.initial_guess(wave, flux)
        lower, upper = model_obj.bounds()

        popt, pcov = curve_fit(
            model_obj.model, wave, flux, p0=p0, sigma=sigma,
            absolute_sigma=sigma is not None, bounds=(lower, upper),
            maxfev=self.maxfev,
        )
        perr = np.sqrt(np.diag(pcov))

        model_flux = model_obj.model(wave, *popt)
        resid = (flux - model_flux)
        if sigma is not None:
            chi2 = float(np.nansum((resid / sigma) ** 2))
        else:
            chi2 = float(np.nansum(resid ** 2))
        dof = max(len(flux) - len(popt), 1)

        return FitResult(
            params=popt, param_names=_get_param_names(model_obj),
            param_errors=perr, covariance=pcov, chi2=chi2, dof=dof,
        )


# ----------------------------------------------------------------------
# MCMC (emcee)
# ----------------------------------------------------------------------
class MCMC(Fitter):
    """
    Bayesian fitting via affine-invariant ensemble MCMC
    (`emcee <https://emcee.readthedocs.io>`_).
    """

    def __init__(self, n_walkers: int = 32, n_steps: int = 2000,
                 n_burn: int = 500, seed: Optional[int] = None):
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_burn = n_burn
        self.seed = seed

    def fit(self, wave, flux, error, model_obj, p0=None) -> FitResult:
        try:
            import emcee
        except ImportError as e:
            raise ImportError(
                "MCMC fitting requires the 'emcee' package: pip install emcee"
            ) from e

        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        sigma = np.asarray(error, dtype=float) if error is not None else np.ones_like(flux)

        if p0 is None:
            p0 = model_obj.initial_guess(wave, flux)
        p0 = np.asarray(p0, dtype=float)
        lower, upper = [np.asarray(b, dtype=float) for b in model_obj.bounds()]

        ndim = len(p0)

        def log_prior(theta):
            if np.any(theta < lower) or np.any(theta > upper):
                return -np.inf
            return 0.0

        def log_likelihood(theta):
            model_flux = model_obj.model(wave, *theta)
            resid = flux - model_flux
            return -0.5 * np.sum((resid / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

        def log_prob(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta)

        rng = np.random.default_rng(self.seed)
        spread = np.maximum(1e-3 * np.abs(p0), 1e-6)
        walkers0 = p0 + spread * rng.normal(size=(self.n_walkers, ndim))
        walkers0 = np.clip(walkers0, lower + 1e-8, upper - 1e-8)

        sampler = emcee.EnsembleSampler(self.n_walkers, ndim, log_prob)
        sampler.run_mcmc(walkers0, self.n_steps, progress=False)

        samples = sampler.get_chain(discard=self.n_burn, flat=True)
        params = np.median(samples, axis=0)
        perr = np.std(samples, axis=0)

        return FitResult(
            params=params, param_names=_get_param_names(model_obj),
            param_errors=perr, samples=samples,
            extra={"sampler": sampler},
        )


def mcmc_fit(wave, flux, error, model):
    """
    Procedural convenience wrapper: run :class:`MCMC` with default
    settings on a model object.

    Parameters
    ----------
    wave, flux, error : ndarray
    model : object
        Any object with ``model``, ``initial_guess``, ``bounds``.

    Returns
    -------
    FitResult
    """
    return MCMC().fit(wave, flux, error, model)


# ----------------------------------------------------------------------
# Nested sampling (dynesty)
# ----------------------------------------------------------------------
class NestedSampling(Fitter):
    """
    Bayesian fitting + evidence estimation via nested sampling
    (`dynesty <https://dynesty.readthedocs.io>`_). Useful for
    rigorous model comparison (e.g. 1 vs 2 Gaussian components).
    """

    def __init__(self, n_live: int = 250, seed: Optional[int] = None):
        self.n_live = n_live
        self.seed = seed

    def fit(self, wave, flux, error, model_obj, p0=None) -> FitResult:
        try:
            import dynesty
        except ImportError as e:
            raise ImportError(
                "Nested sampling requires the 'dynesty' package: pip install dynesty"
            ) from e

        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        sigma = np.asarray(error, dtype=float) if error is not None else np.ones_like(flux)

        lower, upper = [np.asarray(b, dtype=float) for b in model_obj.bounds()]
        ndim = len(lower)
        width = upper - lower

        def prior_transform(u):
            return lower + u * width

        def log_likelihood(theta):
            model_flux = model_obj.model(wave, *theta)
            resid = flux - model_flux
            return -0.5 * np.sum((resid / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

        sampler = dynesty.NestedSampler(
            log_likelihood, prior_transform, ndim, nlive=self.n_live,
        )
        sampler.run_nested(print_progress=False)
        results = sampler.results

        weights = np.exp(results.logwt - results.logz[-1])
        weights /= np.sum(weights)
        samples = results.samples
        mean = np.average(samples, axis=0, weights=weights)
        var = np.average((samples - mean) ** 2, axis=0, weights=weights)
        perr = np.sqrt(var)

        return FitResult(
            params=mean, param_names=_get_param_names(model_obj),
            param_errors=perr, samples=samples,
            log_evidence=float(results.logz[-1]),
            extra={"dynesty_results": results},
        )
