"""
continuum.py
============

AGN continuum components and fitting.

Organized around physical components:

    Continuum
    ├── PowerLaw
    ├── HostGalaxy
    ├── BalmerContinuum
    └── FeII

Each component is a small class exposing:

    * ``param_names``   -- list of free-parameter names
    * ``initial_guess(wave, flux)`` -- reasonable starting values
    * ``bounds()``       -- (lower, upper) bounds for each parameter
    * ``model(wave, *params)`` -- model flux density

A :class:`ContinuumModel` combines an arbitrary subset of components
into one callable used by the fitters in ``fitting.py``.

Free-function wrappers (``fit_powerlaw``, ``fit_host_galaxy``,
``fit_feii``, ``fit_continuum``, ``get_continuum_mask``) are provided
for the simple/procedural workflow described in the module docstring
of ``pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit, nnls
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


# ----------------------------------------------------------------------
# Continuum region masking
# ----------------------------------------------------------------------
def get_continuum_mask(wave, windows):
    """
    Return a boolean mask of "line-free" continuum regions.

    Parameters
    ----------
    wave : ndarray
    windows : list of (float, float)
        Wavelength windows, e.g. ``[(5100, 5130), (4200, 4230)]``,
        known to be free of strong emission lines.

    Returns
    -------
    mask : ndarray of bool
    """
    wave = np.asarray(wave, dtype=float)
    mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in windows:
        mask |= (wave >= lo) & (wave <= hi)
    return mask


# ----------------------------------------------------------------------
# Component base class
# ----------------------------------------------------------------------
class ContinuumComponent:
    """Base class for a single continuum component."""

    name = "component"
    param_names: List[str] = []

    def initial_guess(self, wave, flux):
        raise NotImplementedError

    def bounds(self):
        raise NotImplementedError

    def model(self, wave, *params):
        raise NotImplementedError

    def n_params(self):
        return len(self.param_names)


# ----------------------------------------------------------------------
# Power law
# ----------------------------------------------------------------------
class PowerLaw(ContinuumComponent):
    """
    AGN accretion-disk continuum, modeled as a single power law:

        F(lambda) = A * (lambda / lambda0)^alpha
    """

    name = "powerlaw"
    param_names = ["amplitude", "alpha"]

    def __init__(self, wave0: float = 5100.0):
        self.wave0 = wave0

    def model(self, wave, amplitude, alpha):
        wave = np.asarray(wave, dtype=float)
        return amplitude * (wave / self.wave0) ** alpha

    def initial_guess(self, wave, flux):
        idx = np.argmin(np.abs(np.asarray(wave) - self.wave0))
        amp0 = max(np.asarray(flux)[idx], 1e-3 * np.nanmax(flux))
        return [amp0, -1.5]

    def bounds(self):
        return ([0.0, -5.0], [np.inf, 5.0])


# ----------------------------------------------------------------------
# Host galaxy (stellar population templates)
# ----------------------------------------------------------------------
class HostGalaxy(ContinuumComponent):
    """
    Host-galaxy stellar continuum as a non-negative linear combination
    of stellar-population templates (e.g. simple stellar populations
    or galaxy eigenspectra), optionally with a single velocity
    dispersion broadening common to all templates.

    Parameters
    ----------
    templates : ndarray, shape (n_templates, n_wave)
        Template flux densities, pre-interpolated onto ``wave_grid``.
    wave_grid : ndarray
        Wavelength grid on which ``templates`` are sampled (rest
        frame, Angstrom).
    """

    name = "host_galaxy"

    def __init__(self, templates: np.ndarray, wave_grid: np.ndarray):
        self.templates = np.atleast_2d(np.asarray(templates, dtype=float))
        self.wave_grid = np.asarray(wave_grid, dtype=float)
        self.n_templates = self.templates.shape[0]
        self.param_names = [f"w{i}" for i in range(self.n_templates)]
        self._interpolators = [
            interp1d(self.wave_grid, t, bounds_error=False, fill_value=0.0)
            for t in self.templates
        ]

    def model(self, wave, *weights):
        wave = np.asarray(wave, dtype=float)
        out = np.zeros_like(wave)
        for w, interp in zip(weights, self._interpolators):
            out += w * interp(wave)
        return out

    def initial_guess(self, wave, flux):
        return [0.0] * self.n_templates

    def bounds(self):
        return ([0.0] * self.n_templates, [np.inf] * self.n_templates)

    def fit_weights(self, wave, flux):
        """
        Solve for the non-negative template weights that best
        reproduce ``flux`` via non-negative least squares (NNLS).

        Returns
        -------
        weights : ndarray
        residual_norm : float
        """
        design = np.column_stack([interp(wave) for interp in self._interpolators])
        weights, resid = nnls(design, np.asarray(flux, dtype=float))
        return weights, resid


def fit_host_galaxy(wave, flux, templates):
    """
    Fit stellar population templates to the (continuum-only) flux via
    non-negative least squares.

    Parameters
    ----------
    wave : ndarray
    flux : ndarray
    templates : HostGalaxy or ndarray
        Either a pre-built :class:`HostGalaxy` component, or a raw
        ``(n_templates, n_wave)`` array sampled on ``wave``.

    Returns
    -------
    weights : ndarray
    model_flux : ndarray
    """
    if isinstance(templates, HostGalaxy):
        weights, _ = templates.fit_weights(wave, flux)
        model_flux = templates.model(wave, *weights)
        return weights, model_flux

    templates = np.atleast_2d(np.asarray(templates, dtype=float))
    weights, _ = nnls(templates.T, np.asarray(flux, dtype=float))
    model_flux = templates.T @ weights
    return weights, model_flux


# ----------------------------------------------------------------------
# Balmer continuum
# ----------------------------------------------------------------------
class BalmerContinuum(ContinuumComponent):
    """
    Partially thick Balmer continuum model (Grandi 1982; Dietrich et
    al. 2002 parametrization), important for fitting the AGN continuum
    below the Balmer edge (< 3646 A):

        F(lambda) = A * B_lambda(T_e) * (1 - exp(-tau(lambda))),  lambda < 3646 A
        tau(lambda) = tau0 * (lambda / 3646)^3

    where B_lambda is the Planck function.
    """

    name = "balmer_continuum"
    param_names = ["amplitude", "te", "tau0"]
    edge = 3646.0  # Angstrom

    def model(self, wave, amplitude, te, tau0):
        wave = np.asarray(wave, dtype=float)
        out = np.zeros_like(wave)
        below = wave < self.edge
        lam_cm = wave[below] * 1e-8

        h = 6.62607015e-27
        c = 2.99792458e10
        kb = 1.380649e-16

        planck = (2.0 * h * c ** 2 / lam_cm ** 5) / (
            np.expm1(h * c / (lam_cm * kb * te))
        )
        tau = tau0 * (wave[below] / self.edge) ** 3
        out[below] = amplitude * planck * (1.0 - np.exp(-tau))

        # normalize so "amplitude" is O(flux units) rather than the
        # (huge) natural Planck-function normalization
        if np.any(below) and np.nanmax(out[below]) > 0:
            out[below] /= np.nanmax(planck * (1 - np.exp(-tau)))
            out[below] *= amplitude
        return out

    def initial_guess(self, wave, flux):
        return [np.nanmax(flux) * 0.1, 15000.0, 1.0]

    def bounds(self):
        return ([0.0, 5000.0, 0.01], [np.inf, 30000.0, 5.0])


# ----------------------------------------------------------------------
# Fe II pseudo-continuum
# ----------------------------------------------------------------------
class FeII(ContinuumComponent):
    """
    Optical/UV Fe II pseudo-continuum, modeled as an empirical
    template (e.g. Boroson & Green 1992, Veron-Cetty et al. 2004, or
    Tsuzuki et al. 2006) that is:

      1. Gaussian-broadened to match the AGN's velocity width,
      2. Doppler shifted by a velocity offset,
      3. Scaled by an amplitude.

    Parameters
    ----------
    template_wave, template_flux : ndarray
        Rest-frame Fe II template, in Angstrom / arbitrary flux units.
    """

    name = "feii"
    param_names = ["amplitude", "broadening", "velocity_shift"]

    def __init__(self, template_wave, template_flux):
        self.template_wave = np.asarray(template_wave, dtype=float)
        self.template_flux = np.asarray(template_flux, dtype=float)
        # median pixel size, used to convert km/s broadening to pixels
        self._dwave = np.median(np.diff(self.template_wave))

    def model(self, wave, amplitude, broadening, velocity_shift):
        """
        Parameters
        ----------
        amplitude : float
            Overall scale factor.
        broadening : float
            Extra Gaussian broadening to apply to the template,
            expressed as a velocity sigma in km/s.
        velocity_shift : float
            Doppler velocity shift in km/s.
        """
        c_kms = 2.99792458e5

        # Convert velocity broadening (km/s sigma) to a sigma in
        # Angstrom at the template's mean wavelength, then to pixels.
        mean_wave = np.mean(self.template_wave)
        sigma_ang = broadening / c_kms * mean_wave
        sigma_pix = max(sigma_ang / self._dwave, 1e-3)

        broadened = gaussian_filter1d(self.template_flux, sigma=sigma_pix)

        shifted_wave = self.template_wave * (1.0 + velocity_shift / c_kms)
        interp = interp1d(shifted_wave, broadened, bounds_error=False,
                           fill_value=0.0)
        return amplitude * interp(np.asarray(wave, dtype=float))

    def initial_guess(self, wave, flux):
        return [np.nanmax(flux) * 0.05, 1000.0, 0.0]

    def bounds(self):
        return ([0.0, 100.0, -2000.0], [np.inf, 10000.0, 2000.0])


def fit_feii(wave, flux, template):
    """
    Fit an Fe II template to the flux.

    Parameters
    ----------
    wave, flux : ndarray
    template : FeII or (template_wave, template_flux) tuple

    Returns
    -------
    popt : dict
        {"amplitude":..., "broadening":..., "velocity_shift":...}
    model_flux : ndarray
    """
    if not isinstance(template, FeII):
        template_wave, template_flux = template
        template = FeII(template_wave, template_flux)

    p0 = template.initial_guess(wave, flux)
    lower, upper = template.bounds()
    popt, _ = curve_fit(template.model, wave, flux, p0=p0,
                         bounds=(lower, upper), maxfev=20000)
    model_flux = template.model(wave, *popt)
    result = dict(zip(template.param_names, popt))
    return result, model_flux


# ----------------------------------------------------------------------
# Simple power-law-only fit (procedural convenience function)
# ----------------------------------------------------------------------
def fit_powerlaw(wave, flux, mask):
    """
    Fit an AGN power-law continuum to the masked continuum windows.

    Parameters
    ----------
    wave, flux : ndarray
    mask : ndarray of bool
        Continuum-region mask, e.g. from :func:`get_continuum_mask`.

    Returns
    -------
    popt : dict
        {"amplitude": ..., "alpha": ...}
    model_flux : ndarray
        Power law evaluated over the full input ``wave`` array.
    """
    pl = PowerLaw(wave0=np.median(np.asarray(wave)[mask]))
    p0 = pl.initial_guess(np.asarray(wave)[mask], np.asarray(flux)[mask])
    lower, upper = pl.bounds()
    popt, _ = curve_fit(pl.model, np.asarray(wave)[mask], np.asarray(flux)[mask],
                         p0=p0, bounds=(lower, upper), maxfev=20000)
    model_flux = pl.model(wave, *popt)
    result = dict(zip(pl.param_names, popt))
    result["wave0"] = pl.wave0
    return result, model_flux


# ----------------------------------------------------------------------
# Combined continuum model
# ----------------------------------------------------------------------
@dataclass
class ContinuumModel:
    """
    Combines an arbitrary set of :class:`ContinuumComponent` objects
    into a single additive continuum model, with a joint parameter
    vector for simultaneous fitting.
    """

    components: List[ContinuumComponent] = field(default_factory=list)

    def add(self, component: ContinuumComponent):
        self.components.append(component)
        return self

    @property
    def param_names(self):
        names = []
        for comp in self.components:
            names += [f"{comp.name}_{p}" for p in comp.param_names]
        return names

    def _split_params(self, params):
        out = []
        i = 0
        for comp in self.components:
            n = comp.n_params()
            out.append(params[i:i + n])
            i += n
        return out

    def initial_guess(self, wave, flux):
        guess = []
        for comp in self.components:
            guess += list(comp.initial_guess(wave, flux))
        return guess

    def bounds(self):
        lower, upper = [], []
        for comp in self.components:
            lo, up = comp.bounds()
            lower += list(lo)
            upper += list(up)
        return lower, upper

    def model(self, wave, *params):
        wave = np.asarray(wave, dtype=float)
        total = np.zeros_like(wave)
        for comp, sub_p in zip(self.components, self._split_params(params)):
            total += comp.model(wave, *sub_p)
        return total

    def component_models(self, wave, params):
        """Return a dict of {component_name: model_flux} for each component."""
        out = {}
        for comp, sub_p in zip(self.components, self._split_params(params)):
            out[comp.name] = comp.model(wave, *sub_p)
        return out


def fit_continuum(wave, flux, error=None, components: Optional[List[ContinuumComponent]] = None,
                   continuum_windows: Optional[Sequence[Tuple[float, float]]] = None):
    """
    Fit the full AGN continuum: power law (+ host galaxy, + Fe II,
    + Balmer continuum, if supplied) simultaneously.

    Parameters
    ----------
    wave, flux : ndarray
    error : ndarray, optional
        1-sigma flux errors, used as fit weights if given.
    components : list of ContinuumComponent, optional
        Defaults to ``[PowerLaw()]`` if not given.
    continuum_windows : list of (float, float), optional
        If given, restrict the fit to these line-free windows via
        :func:`get_continuum_mask`; otherwise fit all supplied points.

    Returns
    -------
    continuum_model : ContinuumModel
    popt : ndarray
        Best-fit joint parameter vector.
    model_flux : ndarray
        Combined continuum model evaluated over the full ``wave``
        array.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if components is None:
        components = [PowerLaw()]

    cm = ContinuumModel(components=components)

    if continuum_windows is not None:
        mask = get_continuum_mask(wave, continuum_windows)
    else:
        mask = np.isfinite(flux)

    sigma = None
    if error is not None:
        sigma = np.asarray(error, dtype=float)[mask]

    p0 = cm.initial_guess(wave[mask], flux[mask])
    lower, upper = cm.bounds()

    popt, _ = curve_fit(cm.model, wave[mask], flux[mask], p0=p0,
                         sigma=sigma, absolute_sigma=sigma is not None,
                         bounds=(lower, upper), maxfev=50000)

    model_flux = cm.model(wave, *popt)
    return cm, popt, model_flux
