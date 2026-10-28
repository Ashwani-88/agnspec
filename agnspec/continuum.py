"""
Continuum components: power law, host galaxy and Fe II.

Every component is a small class with the same interface:

* ``param_names``                -- names of the free parameters
* ``initial_guess(wave, flux)``  -- starting values
* ``bounds()``                   -- (lower, upper) limits
* ``model(wave, *params)``       -- model flux density

:class:`ContinuumModel` sums any set of components into one model with a joint
parameter vector, which is what the fitters in ``fitting.py`` work on. The
Balmer pseudo-continuum has its own module, :mod:`agnspec.balmer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import nnls

from .utils import C_KMS


def get_continuum_mask(wave, windows):
    """
    Boolean mask that is True inside any of the given wavelength windows.

    Parameters
    ----------
    wave : ndarray
    windows : list of (float, float)
        Line-free windows, e.g. ``[(4200, 4230), (5080, 5130)]``.
    """
    wave = np.asarray(wave, dtype=float)
    mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in windows:
        mask |= (wave >= lo) & (wave <= hi)
    return mask


class ContinuumComponent:
    """Base class for one continuum component."""

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


class PowerLaw(ContinuumComponent):
    """
    Accretion-disk continuum as a single power law,

        F(lambda) = A * (lambda / lambda0)**alpha

    Parameters
    ----------
    wave0 : float
        Pivot wavelength lambda0 in Angstrom (``amplitude`` is the flux
        density there).
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


class HostGalaxy(ContinuumComponent):
    """
    Host-galaxy starlight as a non-negative combination of templates.

    Parameters
    ----------
    templates : ndarray, shape (n_templates, n_wave)
        Template flux densities sampled on ``wave_grid``.
    wave_grid : ndarray
        Rest-frame wavelength grid of the templates (Angstrom).

    The free parameters are one weight per template, w0, w1, ... The starting
    weights are the non-negative least-squares solution for the data, which
    is also what keeps a bounded fit from sitting on the zero lower limit.
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
        design = np.column_stack([interp(np.asarray(wave, dtype=float))
                                  for interp in self._interpolators])
        weights, _ = nnls(design, np.asarray(flux, dtype=float))
        return list(weights)

    def bounds(self):
        return ([0.0] * self.n_templates, [np.inf] * self.n_templates)


class FeII(ContinuumComponent):
    """
    Fe II pseudo-continuum from an empirical template. The template is
    broadened with a Gaussian, shifted in velocity and scaled.

    Parameters
    ----------
    template_wave, template_flux : ndarray
        Rest-frame Fe II template (Angstrom, arbitrary flux units).

    Notes
    -----
    The broadening is applied as a Gaussian of constant width in Angstrom,
    computed from the velocity sigma at the *mean* wavelength of the template,
    and the template grid is assumed to be evenly spaced. Both are fine for a
    template that spans a narrow region (such as the Park et al. 2022 one, which
    covers the H-beta region). For a template that spans a wide range, crop it
    to the region you are fitting and resample it to an even grid first.
    """

    name = "feii"
    param_names = ["amplitude", "broadening", "velocity_shift"]

    def __init__(self, template_wave, template_flux):
        self.template_wave = np.asarray(template_wave, dtype=float)
        self.template_flux = np.asarray(template_flux, dtype=float)
        # median pixel size, used to turn the km/s broadening into pixels
        self._dwave = np.median(np.diff(self.template_wave))

    def model(self, wave, amplitude, broadening, velocity_shift):
        """
        Parameters
        ----------
        amplitude : float
            Overall scale of the template.
        broadening : float
            Extra Gaussian broadening, as a velocity sigma in km/s.
        velocity_shift : float
            Velocity shift in km/s.
        """
        mean_wave = np.mean(self.template_wave)
        sigma_ang = broadening / C_KMS * mean_wave
        sigma_pix = max(sigma_ang / self._dwave, 1e-3)

        broadened = gaussian_filter1d(self.template_flux, sigma=sigma_pix)

        shifted_wave = self.template_wave * (1.0 + velocity_shift / C_KMS)
        interp = interp1d(shifted_wave, broadened, bounds_error=False, fill_value=0.0)
        return amplitude * interp(np.asarray(wave, dtype=float))

    def initial_guess(self, wave, flux):
        return [np.nanmax(flux) * 0.05, 1000.0, 0.0]

    def bounds(self):
        return ([0.0, 100.0, -2000.0], [np.inf, 10000.0, 2000.0])


@dataclass
class ContinuumModel:
    """
    Sum of several :class:`ContinuumComponent` objects with one joint
    parameter vector. Parameter names are prefixed with the component name,
    e.g. ``powerlaw_alpha``.
    """

    components: List[ContinuumComponent] = field(default_factory=list)

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
        """Dict of {component name: model flux} for a given parameter vector."""
        out = {}
        for comp, sub_p in zip(self.components, self._split_params(params)):
            out[comp.name] = comp.model(wave, *sub_p)
        return out
