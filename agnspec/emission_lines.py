"""
Emission-line complexes.

A ``*Complex`` class groups the Gaussian components that are fitted together
for one spectral region (for example broad + narrow H-beta plus the [OIII]
doublet, with the doublet's centre and flux ratio tied). Like the continuum
components, a complex has ``model``, ``initial_guess``, ``bounds`` and
``free_param_names``, so any fitter in :mod:`agnspec.fitting` can be used on
it.

Available complexes: :class:`HbetaComplex`, :class:`HalphaComplex`,
:class:`MgIIComplex`, :class:`CIVComplex` and :class:`OIIIComplex`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import gaussian, LINE_WAVELENGTHS

OIII_5007_TO_4959 = 2.98     # [OIII] 5007 / 4959 flux ratio
NII_6584_TO_6548 = 2.96      # [NII] 6584 / 6548 flux ratio


def extract_region(wave, flux, xmin, xmax):
    """Return the part of (wave, flux) with ``xmin <= wave <= xmax``."""
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sel = (wave >= xmin) & (wave <= xmax)
    return wave[sel], flux[sel]


class GaussianComponent:
    """
    One named Gaussian inside a line complex (for instance the broad part of H-beta).

    Parameters
    ----------
    name : str
    rest_wave : float
        Rest-frame wavelength in Angstrom.
    kind : {"broad", "narrow"}
        Selects the default starting values and sigma bounds.
    tie_center_to : GaussianComponent, optional
        Keep this line at a fixed rest-wavelength offset from another one
        (e.g. [OIII] 4959 tied to [OIII] 5007).
    flux_ratio_to : (GaussianComponent, float), optional
        Force this line's amplitude to ``ratio`` times the other line's
        amplitude (e.g. 4959 = 5007 / 2.98).
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
    Base class for a group of emission lines fitted together. Subclasses fill
    ``self.components`` with :class:`GaussianComponent` objects in ``__init__``.

    All Gaussian widths and centres are in Angstrom, amplitudes in flux
    density units.
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
        """(amplitude, center, sigma) of one component, given the free parameters."""
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
        span = 50.0     # allowed shift of a line centre, Angstrom
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
        Resolved (amplitude, center, sigma) of every component for a fitted
        parameter vector, keyed by component name.
        """
        free_values = dict(zip(self.free_param_names, params))
        return {comp.name: self._resolve(free_values, comp) for comp in self.components}


def _oiii_doublet(suffix="", kind="narrow"):
    """[OIII] 5007 plus 4959, with 4959 tied to 5007 in centre and flux."""
    o5007 = GaussianComponent(f"oiii_5007{suffix}", LINE_WAVELENGTHS["OIII_5007"], kind)
    o4959 = GaussianComponent(
        f"oiii_4959{suffix}", LINE_WAVELENGTHS["OIII_4959"], kind,
        tie_center_to=o5007, flux_ratio_to=(o5007, 1.0 / OIII_5007_TO_4959),
    )
    return [o5007, o4959]


class HbetaComplex(LineComplex):
    """Narrow H-beta, optional broad H-beta, optional [OIII] 4959,5007 doublet."""

    name = "hbeta_complex"

    def __init__(self, include_broad=True, include_oiii=True):
        super().__init__()
        self.components.append(
            GaussianComponent("hbeta_narrow", LINE_WAVELENGTHS["Hbeta"], "narrow"))

        if include_broad:
            self.components.append(
                GaussianComponent("hbeta_broad", LINE_WAVELENGTHS["Hbeta"], "broad"))

        if include_oiii:
            self.components += _oiii_doublet()


class HalphaComplex(LineComplex):
    """
    Narrow H-alpha, optional broad H-alpha, optional [NII] 6548,6584 doublet
    (fixed ratio, tied centres) and optional [SII] 6716,6731 pair (tied centres).
    """

    name = "halpha_complex"

    def __init__(self, include_broad=True, include_nii=True, include_sii=True):
        super().__init__()
        self.components.append(
            GaussianComponent("halpha_narrow", LINE_WAVELENGTHS["Halpha"], "narrow"))

        if include_broad:
            self.components.append(
                GaussianComponent("halpha_broad", LINE_WAVELENGTHS["Halpha"], "broad"))

        if include_nii:
            nii_6584 = GaussianComponent("nii_6584", LINE_WAVELENGTHS["NII_6584"], "narrow")
            nii_6548 = GaussianComponent(
                "nii_6548", LINE_WAVELENGTHS["NII_6548"], "narrow",
                tie_center_to=nii_6584, flux_ratio_to=(nii_6584, 1.0 / NII_6584_TO_6548),
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
    """Mg II 2798: broad Gaussian plus an optional narrow one."""

    name = "mgii_complex"

    def __init__(self, include_narrow=True):
        super().__init__()
        self.components.append(
            GaussianComponent("mgii_broad", LINE_WAVELENGTHS["MgII"], "broad"))
        if include_narrow:
            self.components.append(
                GaussianComponent("mgii_narrow", LINE_WAVELENGTHS["MgII"], "narrow"))


class CIVComplex(LineComplex):
    """C IV 1549: broad Gaussian plus an optional second (wind) Gaussian."""

    name = "civ_complex"

    def __init__(self, include_wind=False):
        super().__init__()
        self.components.append(
            GaussianComponent("civ_broad", LINE_WAVELENGTHS["CIV"], "broad"))
        if include_wind:
            self.components.append(
                GaussianComponent("civ_wind", LINE_WAVELENGTHS["CIV"], "broad"))


class OIIIComplex(LineComplex):
    """
    [OIII] 4959,5007 on its own (fixed flux ratio, tied centres), with an
    optional second, broader doublet for an outflow wing.
    """

    name = "oiii_complex"

    def __init__(self, include_wing=False):
        super().__init__()
        self.components += _oiii_doublet()
        if include_wing:
            self.components += _oiii_doublet(suffix="_wing", kind="broad")


LINE_COMPLEXES = {
    "Hbeta": HbetaComplex,
    "Halpha": HalphaComplex,
    "MgII": MgIIComplex,
    "CIV": CIVComplex,
    "OIII": OIIIComplex,
}
