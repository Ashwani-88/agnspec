"""
:class:`AGNModel` ties a continuum model, any number of emission-line
complexes and a fitter together, and offers short-cuts for measuring and
plotting the result.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .continuum import ContinuumComponent, ContinuumModel, PowerLaw
from .emission_lines import LineComplex, LINE_COMPLEXES
from .fitting import Fitter, LeastSquares, FitResult
from . import measurements as meas
from . import plotting as plots


class AGNModel:
    """
    A continuum model plus one or more emission-line complexes, fitted with
    whichever fitter you choose.

    Parameters
    ----------
    continuum_components : list of ContinuumComponent, optional
        Defaults to ``[PowerLaw()]``.
    line_complexes : dict of {name: LineComplex}, optional
        E.g. ``{"Hbeta": HbetaComplex()}``. Complexes can also be
        added later via :meth:`add_line_complex`.
    fitter : Fitter, optional
        Defaults to :class:`fitting.LeastSquares`.
    """

    def __init__(self, continuum_components: Optional[List[ContinuumComponent]] = None,
                 line_complexes: Optional[Dict[str, LineComplex]] = None,
                 fitter: Optional[Fitter] = None):
        self.continuum = ContinuumModel(
            components=continuum_components or [PowerLaw()]
        )
        self.line_complexes: Dict[str, LineComplex] = dict(line_complexes or {})
        self.fitter = fitter or LeastSquares()

        # populated after fitting
        self.continuum_result: Optional[FitResult] = None
        self.line_results: Dict[str, FitResult] = {}
        self.wave = None
        self.flux = None
        self.error = None

    def add_line_complex(self, name: str, complex_obj: Optional[LineComplex] = None, **kwargs):
        """
        Attach a named emission-line complex. If ``complex_obj`` is
        not given, ``name`` is looked up in
        :data:`emission_lines.LINE_COMPLEXES` (e.g. "Hbeta",
        "Halpha", "MgII", "CIV", "OIII") and instantiated with
        ``**kwargs``.
        """
        if complex_obj is None:
            if name not in LINE_COMPLEXES:
                raise KeyError(f"Unknown line complex '{name}'. "
                                f"Known: {list(LINE_COMPLEXES)}")
            complex_obj = LINE_COMPLEXES[name](**kwargs)
        self.line_complexes[name] = complex_obj
        return self

    def fit_continuum(self, wave, flux, error=None, continuum_windows=None):
        """Fit the continuum model to (typically line-masked) data."""
        from .continuum import get_continuum_mask

        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        self.wave, self.flux, self.error = wave, flux, error

        if continuum_windows is not None:
            mask = get_continuum_mask(wave, continuum_windows)
        else:
            mask = np.isfinite(flux)

        err_fit = np.asarray(error, dtype=float)[mask] if error is not None else None
        result = self.fitter.fit(wave[mask], flux[mask], err_fit, self.continuum)
        self.continuum_result = result
        return result

    def continuum_flux(self, wave=None):
        """Evaluate the best-fit continuum model over ``wave`` (default: stored wave)."""
        if self.continuum_result is None:
            raise RuntimeError("Continuum has not been fit yet; call fit_continuum() first.")
        wave = wave if wave is not None else self.wave
        return self.continuum.model(wave, *self.continuum_result.params)

    def fit_line(self, name, wave, flux, error=None):
        """
        Fit a previously-added line complex to continuum-subtracted
        flux over the wavelength range ``wave``.
        """
        if name not in self.line_complexes:
            raise KeyError(f"Line complex '{name}' has not been added; "
                            f"call add_line_complex('{name}') first.")
        complex_obj = self.line_complexes[name]
        result = self.fitter.fit(wave, flux, error, complex_obj)
        self.line_results[name] = result
        return result

    def fit_all_lines(self, wave, flux, error=None, windows: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Fit every added line complex, optionally restricting each fit
        to a wavelength window ``windows[name] = (xmin, xmax)``.
        """
        from .emission_lines import extract_region

        results = {}
        for name, complex_obj in self.line_complexes.items():
            if windows and name in windows:
                xmin, xmax = windows[name]
                w, f = extract_region(wave, flux, xmin, xmax)
                e = None
                if error is not None:
                    _, e = extract_region(wave, error, xmin, xmax)
            else:
                w, f, e = wave, flux, error
            results[name] = self.fit_line(name, w, f, e)
        return results

    def measure_line(self, name):
        """
        Flux, FWHM (Angstrom), velocity dispersion and velocity shift (km/s)
        of every Gaussian in a fitted line complex.

        Returns
        -------
        dict of {component_name: {"flux": ..., "fwhm": ..., ...}}
        """
        if name not in self.line_results:
            raise RuntimeError(f"Line complex '{name}' has not been fit yet.")
        complex_obj = self.line_complexes[name]
        result = self.line_results[name]
        resolved = complex_obj.component_fluxes(result.params)

        out = {}
        for comp_name, (amp, cen, sig) in resolved.items():
            comp = next(c for c in complex_obj.components if c.name == comp_name)
            gparams = {"amplitude": amp, "center": cen, "sigma": sig}
            out[comp_name] = {
                "amplitude": amp,
                "center": cen,
                "sigma": sig,
                "flux": meas.line_flux(gparams),
                "fwhm": meas.fwhm(gparams),
                "sigma_velocity": meas.sigma_velocity(gparams, rest_wave=comp.rest_wave),
                "velocity_shift": meas.velocity_shift(cen, comp.rest_wave),
            }
        return out

    def plot_continuum(self, ax=None):
        model_flux = self.continuum_flux()
        components = self.continuum.component_models(self.wave, self.continuum_result.params)
        return plots.plot_continuum_fit(self.wave, self.flux, model_flux,
                                         components=components, ax=ax)

    def plot_line(self, name, wave, flux, continuum=None, ax=None):
        if name not in self.line_results:
            raise RuntimeError(f"Line complex '{name}' has not been fit yet.")
        complex_obj = self.line_complexes[name]
        result = self.line_results[name]
        model_flux = complex_obj.model(wave, *result.params)
        resolved = complex_obj.component_fluxes(result.params)
        return plots.plot_line_fit(wave, flux, model_flux, components=resolved,
                                    continuum=continuum, ax=ax)

    def summary(self):
        """Return a flat dict of all fitted parameters, ready for saving."""
        out = {}
        if self.continuum_result is not None:
            out["continuum"] = self.continuum_result.as_dict()
        for name, result in self.line_results.items():
            out[f"line_{name}"] = result.as_dict()
        return out
