"""
emission_lines.py
==================

Emission-line region extraction and Gaussian-component fitting,
organized around physical line complexes:

    EmissionLines
    ├── HbetaComplex
    ├── HalphaComplex
    ├── MgIIComplex
    ├── CIVComplex
    └── OIIIComplex

Each ``*Complex`` class defines the set of Gaussian sub-components
that make up a spectral region (e.g. broad + narrow Hbeta, the
[OIII] doublet with a fixed 3:1 flux ratio and common kinematics,
etc.) and knows how to build an initial guess, bounds, and a combined
model, exactly like the ``ContinuumComponent`` classes in
``continuum.py``. This lets :class:`fitting.LeastSquares`,
:class:`fitting.MCMC`, and :class:`fitting.NestedSampling` operate on
either continuum or line objects interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson

from .utils import gaussian, multi_gaussian, LINE_WAVELENGTHS, fwhm_to_sigma


# ----------------------------------------------------------------------
# Region extraction
# ----------------------------------------------------------------------
def extract_region(wave, flux, xmin, xmax):
    """
    Extract the sub-array of a spectrum within ``[xmin, xmax]``.

    Parameters
    ----------
    wave, flux : ndarray
    xmin, xmax : float

    Returns
    -------
    wave_region, flux_region : ndarray
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sel = (wave >= xmin) & (wave <= xmax)
    return wave[sel], flux[sel]


# ----------------------------------------------------------------------
# Generic Gaussian fitting (procedural convenience functions)
# ----------------------------------------------------------------------
def fit_gaussian(wave, flux, p0=None):
    """
    Fit a single Gaussian (+ optional local linear background) to a
    line region.

    Parameters
    ----------
    wave, flux : ndarray
    p0 : sequence, optional
        Initial guess ``[amplitude, center, sigma]``. If not given, a
        reasonable guess is derived from the data.

    Returns
    -------
    popt : dict
        {"amplitude":..., "center":..., "sigma":...}
    model_flux : ndarray
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if p0 is None:
        i_max = np.argmax(flux)
        p0 = [flux[i_max], wave[i_max], (wave.max() - wave.min()) / 10.0]

    lower = [0.0, wave.min(), 0.01]
    upper = [np.inf, wave.max(), (wave.max() - wave.min())]

    popt, _ = curve_fit(gaussian, wave, flux, p0=p0, bounds=(lower, upper),
                         maxfev=20000)
    model_flux = gaussian(wave, *popt)
    result = dict(zip(["amplitude", "center", "sigma"], popt))
    return result, model_flux


def fit_multiple_gaussians(wave, flux, n_components, p0=None):
    """
    Fit ``n_components`` Gaussians simultaneously to a line region.

    Parameters
    ----------
    wave, flux : ndarray
    n_components : int
    p0 : sequence, optional
        Flat initial-guess vector of length ``3 * n_components``:
        ``[amp1, cen1, sig1, amp2, cen2, sig2, ...]``. If not given,
        components are seeded at evenly spaced positions across the
        region with equal amplitude fractions of the peak flux.

    Returns
    -------
    popt : ndarray, shape (n_components, 3)
        Best-fit (amplitude, center, sigma) for each component.
    model_flux : ndarray
        Combined (summed) model.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if p0 is None:
        centers0 = np.linspace(wave.min(), wave.max(), n_components + 2)[1:-1]
        p0 = []
        for c in centers0:
            p0 += [np.nanmax(flux) / n_components, c,
                   (wave.max() - wave.min()) / (4 * n_components)]

    lower, upper = [], []
    for _ in range(n_components):
        lower += [0.0, wave.min(), 0.01]
        upper += [np.inf, wave.max(), (wave.max() - wave.min())]

    popt, _ = curve_fit(multi_gaussian, wave, flux, p0=p0,
                         bounds=(lower, upper), maxfev=50000)
    model_flux = multi_gaussian(wave, *popt)
    popt = np.asarray(popt).reshape(n_components, 3)
    return popt, model_flux


# ----------------------------------------------------------------------
# Gaussian component object (used inside *Complex classes)
# ----------------------------------------------------------------------
class GaussianComponent:
    """
    A single named Gaussian sub-component of a line complex (e.g. the
    "broad" or "narrow" part of Hbeta).

    Parameters
    ----------
    name : str
    rest_wave : float
        Rest-frame wavelength (Angstrom).
    kind : str
        "broad" or "narrow"; used only for bookkeeping / plotting.
    tie_center_to : GaussianComponent, optional
        If given, this component's center is forced to track another
        component's center via a fixed rest-wavelength offset (e.g.
        the [OIII] 4959 line tied to [OIII] 5007).
    flux_ratio_to : Tuple[GaussianComponent, float], optional
        If given as ``(other_component, ratio)``, this component's
        amplitude is forced to equal ``ratio`` times the other
        component's amplitude (e.g. [OIII] 4959 = 1/3 x [OIII] 5007).
    """

    def __init__(self, name, rest_wave, kind="broad",
                 tie_center_to: Optional["GaussianComponent"] = None,
                 flux_ratio_to: Optional[Tuple["GaussianComponent", float]] = None):
        self.name = name
        self.rest_wave = rest_wave
        self.kind = kind
        self.tie_center_to = tie_center_to
        self.flux_ratio_to = flux_ratio_to

    @property
    def is_free(self):
        return self.tie_center_to is None and self.flux_ratio_to is None

    @property
    def free_param_names(self):
        if self.flux_ratio_to is not None and self.tie_center_to is not None:
            return [f"{self.name}_sigma"]
        if self.flux_ratio_to is not None:
            return [f"{self.name}_center", f"{self.name}_sigma"]
        if self.tie_center_to is not None:
            return [f"{self.name}_amplitude", f"{self.name}_sigma"]
        return [f"{self.name}_amplitude", f"{self.name}_center", f"{self.name}_sigma"]


class LineComplex:
    """
    Base class for a physically-motivated group of emission lines
    fitted together (e.g. Hbeta + [OIII] doublet).

    Subclasses populate ``self.components`` with :class:`GaussianComponent`
    objects (some tied to others) in ``__init__``.
    """

    name = "line_complex"

    def __init__(self):
        self.components: List[GaussianComponent] = []

    @property
    def free_param_names(self):
        names = []
        for c in self.components:
            names += c.free_param_names
        return names

    def _resolve(self, free_values: Dict[str, float], comp: GaussianComponent):
        """Resolve (amplitude, center, sigma) for one component given
        the dict of currently-fitted free parameter values."""
        sigma = free_values[f"{comp.name}_sigma"]

        if comp.tie_center_to is not None:
            ref = comp.tie_center_to
            offset = comp.rest_wave - ref.rest_wave
            center = free_values.get(f"{ref.name}_center") + offset
        else:
            center = free_values[f"{comp.name}_center"]

        if comp.flux_ratio_to is not None:
            ref_comp, ratio = comp.flux_ratio_to
            amplitude = ratio * free_values[f"{ref_comp.name}_amplitude"]
        else:
            amplitude = free_values[f"{comp.name}_amplitude"]

        return amplitude, center, sigma

    def model(self, wave, *params):
        wave = np.asarray(wave, dtype=float)
        free_values = dict(zip(self.free_param_names, params))
        total = np.zeros_like(wave)
        for comp in self.components:
            amp, cen, sig = self._resolve(free_values, comp)
            total += gaussian(wave, amp, cen, sig)
        return total

    def initial_guess(self, wave, flux):
        guess = []
        peak = np.nanmax(flux) if len(flux) else 1.0
        span = (np.max(wave) - np.min(wave)) if len(wave) else 100.0
        for comp in self.components:
            for pname in comp.free_param_names:
                if pname.endswith("_amplitude"):
                    guess.append(0.3 * peak if comp.kind == "narrow" else 0.5 * peak)
                elif pname.endswith("_center"):
                    guess.append(comp.rest_wave)
                elif pname.endswith("_sigma"):
                    guess.append(3.0 if comp.kind == "narrow" else 15.0)
        return guess

    def bounds(self):
        lower, upper = [], []
        span = 50.0
        for comp in self.components:
            for pname in comp.free_param_names:
                if pname.endswith("_amplitude"):
                    lower.append(0.0)
                    upper.append(np.inf)
                elif pname.endswith("_center"):
                    lower.append(comp.rest_wave - span)
                    upper.append(comp.rest_wave + span)
                elif pname.endswith("_sigma"):
                    lower.append(0.1)
                    upper.append(2000.0 if comp.kind == "broad" else 20.0)
        return lower, upper

    def component_fluxes(self, params):
        """
        Given a best-fit joint parameter vector, return the resolved
        (amplitude, center, sigma) for every sub-component, keyed by
        component name.
        """
        free_values = dict(zip(self.free_param_names, params))
        out = {}
        for comp in self.components:
            out[comp.name] = self._resolve(free_values, comp)
        return out


class HbetaComplex(LineComplex):
    """
    H-beta region: broad Hbeta + narrow Hbeta + [OIII]4959,5007
    doublet (tied kinematics, fixed 3:1 flux ratio).
    """

    name = "hbeta_complex"

    def __init__(self, include_broad=True, include_oiii=True):
        super().__init__()
        hb_n = GaussianComponent("hbeta_narrow", LINE_WAVELENGTHS["Hbeta"], "narrow")
        self.components.append(hb_n)

        if include_broad:
            hb_b = GaussianComponent("hbeta_broad", LINE_WAVELENGTHS["Hbeta"], "broad")
            self.components.append(hb_b)

        if include_oiii:
            oiii_5007 = GaussianComponent("oiii_5007", LINE_WAVELENGTHS["OIII_5007"], "narrow")
            oiii_4959 = GaussianComponent(
                "oiii_4959", LINE_WAVELENGTHS["OIII_4959"], "narrow",
                tie_center_to=oiii_5007, flux_ratio_to=(oiii_5007, 1.0 / 2.98),
            )
            self.components += [oiii_5007, oiii_4959]


class HalphaComplex(LineComplex):
    """
    H-alpha region: broad Halpha + narrow Halpha + [NII]6548,6583
    doublet (fixed 1:2.96 ratio, tied kinematics) + [SII]6716,6731.
    """

    name = "halpha_complex"

    def __init__(self, include_broad=True, include_nii=True, include_sii=True):
        super().__init__()
        ha_n = GaussianComponent("halpha_narrow", LINE_WAVELENGTHS["Halpha"], "narrow")
        self.components.append(ha_n)

        if include_broad:
            ha_b = GaussianComponent("halpha_broad", LINE_WAVELENGTHS["Halpha"], "broad")
            self.components.append(ha_b)

        if include_nii:
            nii_6584 = GaussianComponent("nii_6584", LINE_WAVELENGTHS["NII_6584"], "narrow")
            nii_6548 = GaussianComponent(
                "nii_6548", LINE_WAVELENGTHS["NII_6548"], "narrow",
                tie_center_to=nii_6584, flux_ratio_to=(nii_6584, 1.0 / 2.96),
            )
            self.components += [nii_6584, nii_6548]

        if include_sii:
            sii_6716 = GaussianComponent("sii_6716", LINE_WAVELENGTHS["SII_6716"], "narrow")
            sii_6731 = GaussianComponent(
                "sii_6731", LINE_WAVELENGTHS["SII_6731"], "narrow",
                tie_center_to=sii_6716,
            )
            self.components += [sii_6716, sii_6731]


class MgIIComplex(LineComplex):
    """MgII 2798 region: broad + narrow MgII."""

    name = "mgii_complex"

    def __init__(self, include_narrow=True):
        super().__init__()
        self.components.append(
            GaussianComponent("mgii_broad", LINE_WAVELENGTHS["MgII"], "broad")
        )
        if include_narrow:
            self.components.append(
                GaussianComponent("mgii_narrow", LINE_WAVELENGTHS["MgII"], "narrow")
            )


class CIVComplex(LineComplex):
    """CIV 1549 region: broad (+ optional blueshifted-wind) component."""

    name = "civ_complex"

    def __init__(self, include_wind=False):
        super().__init__()
        self.components.append(
            GaussianComponent("civ_broad", LINE_WAVELENGTHS["CIV"], "broad")
        )
        if include_wind:
            self.components.append(
                GaussianComponent("civ_wind", LINE_WAVELENGTHS["CIV"], "broad")
            )


class OIIIComplex(LineComplex):
    """
    Standalone [OIII] 4959,5007 doublet (e.g. for cases where Hbeta
    is not being modeled at all), fixed 3:1 flux ratio, tied
    kinematics, with an optional second ("wing"/outflow) component
    tied similarly.
    """

    name = "oiii_complex"

    def __init__(self, include_wing=False):
        super().__init__()
        oiii_5007 = GaussianComponent("oiii_5007", LINE_WAVELENGTHS["OIII_5007"], "narrow")
        oiii_4959 = GaussianComponent(
            "oiii_4959", LINE_WAVELENGTHS["OIII_4959"], "narrow",
            tie_center_to=oiii_5007, flux_ratio_to=(oiii_5007, 1.0 / 2.98),
        )
        self.components += [oiii_5007, oiii_4959]

        if include_wing:
            oiii_5007_w = GaussianComponent("oiii_5007_wing", LINE_WAVELENGTHS["OIII_5007"], "broad")
            oiii_4959_w = GaussianComponent(
                "oiii_4959_wing", LINE_WAVELENGTHS["OIII_4959"], "broad",
                tie_center_to=oiii_5007_w, flux_ratio_to=(oiii_5007_w, 1.0 / 2.98),
            )
            self.components += [oiii_5007_w, oiii_4959_w]


LINE_COMPLEXES = {
    "Hbeta": HbetaComplex,
    "Halpha": HalphaComplex,
    "MgII": MgIIComplex,
    "CIV": CIVComplex,
    "OIII": OIIIComplex,
}
