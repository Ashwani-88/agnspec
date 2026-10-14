"""
plotting.py
============

Diagnostic plots: raw spectrum, continuum fits, and emission-line
fits (with individual sub-components overplotted).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt



# Default y-axis unit string, used when the caller doesn't specify one
# explicitly. This is only a *guess* -- real flux-calibrated spectra
# vary a lot in normalization (e.g. SDSS spectra are typically in
# units of 1e-17 erg/s/cm^2/Angstrom, not erg/s/cm^2/Angstrom). Pass
# `flux_unit=...` to any plotting function below to override it, or
# use `flux_unit_from_header()` to pull it from the FITS header
# returned by `io.read_spectrum`.
DEFAULT_FLUX_UNIT = r"erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$"


def flux_unit_from_header(header, default=DEFAULT_FLUX_UNIT):
    """
    Build a y-axis unit string from a spectrum's FITS header, falling
    back to ``default`` if no relevant keyword is found.

    Looks for a ``BUNIT`` keyword (and, failing that, common SDSS-style
    ``FLUXUNIT``/``BSCALE`` conventions). The raw header string is
    returned as-is (not LaTeX-formatted), since header unit strings
    are inconsistent enough that guessing a "pretty" version is more
    likely to mislead than help -- pass an explicit ``flux_unit=``
    string to any plotting function if you want LaTeX formatting.

    Parameters
    ----------
    header : dict
    default : str

    Returns
    -------
    str
    """
    if not header:
        return default
    for key in ("BUNIT", "FLUXUNIT", "TUNIT2"):
        val = header.get(key)
        if val:
            return str(val).strip()
    return default


def _ylabel(flux_unit):
    flux_unit = flux_unit if flux_unit is not None else DEFAULT_FLUX_UNIT
    return f"Flux density [{flux_unit}]"
    
# Presentation defaults for each stage of the preprocessing pipeline.
# "stage" controls the x-axis label (observed vs. rest frame) and the
# default line label/color, so plot_spectrum() can be reused as-is for
# raw, extinction-corrected, and rest-frame spectra -- including
# overlaying several stages on the same Axes for a quick QA look.
_STAGE_INFO = {
    "raw": {
        "xlabel": r"Observed-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Raw spectrum",
        "color": "0.15",
    },
    "observed": {  # alias of "raw"
        "xlabel": r"Observed-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Observed spectrum",
        "color": "0.15",
    },
    "dered": {
        "xlabel": r"Observed-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Extinction-corrected spectrum",
        "color": "steelblue",
    },
    "extinction_corrected": {  # alias of "dered"
        "xlabel": r"Observed-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Extinction-corrected spectrum",
        "color": "steelblue",
    },
    "smooth": {
        "xlabel": r"Rest-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Smoothed spectrum",
        "color": "crimson",
    },
    "rest": {  # alias of "rest"
        "xlabel": r"Rest-frame Wavelength [$\mathrm{\AA}$]",
        "label": "rest-frame spectrum",
        "color": "crimson",
    },
}


def plot_spectrum(wave, flux, error=None, ax=None, stage="raw",
                   label=None, color=None, flux_unit=None,
                   save=None, dpi=150, **kwargs):
    """
    Plot a spectrum at any stage of preprocessing: raw (observed),
    extinction-corrected, or rest-frame.

    Parameters
    ----------
    wave, flux : ndarray
    error : ndarray, optional
        If given, plotted as a shaded 1-sigma band.
    ax : matplotlib.axes.Axes, optional
        Pass an existing Axes to overlay several stages on one plot
        (e.g. raw + extinction-corrected + rest-frame together).
    stage : str
        One of ``"raw"``/``"observed"``, ``"dered"``/``"extinction_corrected"``,
        or ``"rest"``/``"rest_frame"``. Controls the default line
        label/color and the x-axis label (observed- vs rest-frame
        wavelength). Unrecognized values fall back to "raw" styling.
    label : str, optional
        Overrides the default legend label for this ``stage``.
    color : str, optional
        Overrides the default line color for this ``stage``.
    flux_unit : str, optional
        Unit string shown in the y-axis label, e.g.
        ``r"10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\\mathrm{\\AA}^{-1}$"``
        for typical SDSS-scaled spectra, or the output of
        :func:`flux_unit_from_header`. Defaults to
        :data:`DEFAULT_FLUX_UNIT` if not given.
    **kwargs
        Passed through to ``ax.plot``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    Plot each stage on its own figure, with the real flux unit for
    SDSS-style spectra::

        unit = r"10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\\mathrm{\\AA}^{-1}$"
        plot_spectrum(wave_obs, flux_obs, stage="raw", flux_unit=unit)
        plot_spectrum(wave_obs, flux_dered, stage="extinction_corrected", flux_unit=unit)
        plot_spectrum(wave_rest, flux_rest, error=error_rest, stage="rest_frame", flux_unit=unit)

    Or pull the unit straight from the file header::

        wave, flux, error, header = read_spectrum("rm160.fits")
        plot_spectrum(wave, flux, stage="raw", flux_unit=flux_unit_from_header(header))

    Overlay all three stages for a quick before/after comparison::

        ax = plot_spectrum(wave_obs, flux_obs, stage="raw", flux_unit=unit)
        plot_spectrum(wave_obs, flux_dered, ax=ax, stage="extinction_corrected", flux_unit=unit)
        plot_spectrum(wave_rest, flux_rest, ax=ax, stage="rest_frame", flux_unit=unit)
    """
    info = _STAGE_INFO.get(stage, _STAGE_INFO["raw"])
    label = label if label is not None else info["label"]
    color = color if color is not None else info["color"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(wave, flux, lw=1.0, color=color, label=label, **kwargs)
    if error is not None:
        ax.fill_between(wave, flux - error, flux + error, color=color, alpha=0.25,
                         step=None, linewidth=0)

    ax.set_xlabel(info["xlabel"])
    ax.set_ylabel(_ylabel(flux_unit))
    ax.legend(loc="best", frameon=False)
    if save is not None:
        ax.figure.savefig(save, dpi=dpi, bbox_inches="tight")

    return ax


def compare_spectra(stages, ax=None):
    """
    Overlay multiple pipeline stages of the *same* spectrum on one
    Axes, e.g. to visually confirm dereddening and the rest-frame
    shift did what you expect.

    Parameters
    ----------
    stages : list of dict
        Each dict describes one call to :func:`plot_spectrum`, e.g.::

            [
                {"wave": wave_obs,  "flux": flux_obs,   "stage": "raw"},
                {"wave": wave_obs,  "flux": flux_dered, "stage": "extinction_corrected"},
                {"wave": wave_rest, "flux": flux_rest,  "stage": "rest_frame", "error": error_rest},
            ]

    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    for s in stages:
        s = dict(s)
        wave = s.pop("wave")
        flux = s.pop("flux")
        ax = plot_spectrum(wave, flux, ax=ax, **s)
    return ax


def plot_continuum_fit(wave, flux, model, components: Optional[Dict[str, np.ndarray]] = None,
                        ax=None, mask=None):
    """
    Plot the observed spectrum together with the fitted continuum
    model (and, optionally, its individual components).

    Parameters
    ----------
    wave, flux : ndarray
    model : ndarray
        Combined continuum model flux, evaluated at ``wave``.
    components : dict, optional
        {component_name: model_flux_array}, e.g. from
        :meth:`continuum.ContinuumModel.component_models`.
    ax : matplotlib.axes.Axes, optional
    mask : ndarray of bool, optional
        Continuum-fitting mask; highlighted if given.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(wave, flux, lw=1.0, color="0.15", label="data")
    ax.plot(wave, model, lw=1.8, color="crimson", label="continuum (total)")

    if components:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(components)))
        for (name, comp_flux), c in zip(components.items(), colors):
            ax.plot(wave, comp_flux, lw=1.2, ls="--", color=c, label=name)

    if mask is not None:
        ax.plot(np.asarray(wave)[mask], np.asarray(flux)[mask], ".", ms=3,
                color="dodgerblue", label="continuum windows", zorder=5)

    ax.set_xlabel(r"Rest-frame Wavelength [$\mathrm{\AA}$]")
    ax.set_ylabel(r"Flux density [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]")
    ax.legend(loc="best", frameon=False, fontsize=8)
    return ax


def plot_line_fit(wave, flux, model, components: Optional[Dict[str, np.ndarray]] = None,
                   continuum=None, ax=None, residual_panel=True):
    """
    Plot an emission-line-region fit: data, combined model, and
    (optionally) each Gaussian sub-component, with a residual panel.

    Parameters
    ----------
    wave, flux : ndarray
        Continuum-subtracted flux (or set ``continuum`` to overplot
        pre-subtraction data with the continuum added back in).
    model : ndarray
        Combined line-complex model flux.
    components : dict, optional
        {component_name: (amplitude, center, sigma)} or
        {component_name: model_flux_array}.
    continuum : ndarray, optional
        If given, both data and model are plotted with the continuum
        added back in (i.e. in "total flux" space).
    ax : matplotlib.axes.Axes, optional
    residual_panel : bool
        Whether to add a residual sub-panel below the main plot.

    Returns
    -------
    fig, axes
    """
    from .utils import gaussian

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    model = np.asarray(model, dtype=float)
    cont = np.asarray(continuum, dtype=float) if continuum is not None else np.zeros_like(wave)

    if ax is None:
        if residual_panel:
            fig, axes = plt.subplots(
                2, 1, figsize=(8, 6), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            )
            ax_main, ax_resid = axes
        else:
            fig, ax_main = plt.subplots(figsize=(8, 4.5))
            axes = ax_main
    else:
        ax_main = ax
        fig = ax_main.figure
        axes = ax_main

    ax_main.plot(wave, flux + cont, lw=1.0, color="0.15", label="data")
    ax_main.plot(wave, model + cont, lw=1.8, color="crimson", label="model (total)")

    if components:
        colors = plt.cm.plasma(np.linspace(0, 0.85, len(components)))
        for (name, val), c in zip(components.items(), colors):
            if isinstance(val, tuple) and len(val) == 3:
                amp, cen, sig = val
                comp_flux = gaussian(wave, amp, cen, sig)
            else:
                comp_flux = np.asarray(val)
            ax_main.plot(wave, comp_flux + cont, lw=1.0, ls="--", color=c, label=name)

    ax_main.set_ylabel(r"Flux density [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]")
    ax_main.legend(loc="best", frameon=False, fontsize=8)

    if residual_panel:
        ax_resid.axhline(0, color="0.5", lw=0.8)
        ax_resid.plot(wave, flux - model, lw=1.0, color="0.3")
        ax_resid.set_ylabel("Residual")
        ax_resid.set_xlabel(r"Rest-frame Wavelength [$\mathrm{\AA}$]")
    else:
        ax_main.set_xlabel(r"Rest-frame Wavelength [$\mathrm{\AA}$]")

    return fig, axes
    
    
def plot_continuum_windows(wave, flux, mask, error=None, ax=None,
                            flux_unit=None, stage="rest", save=None):
    """
    Plot the spectrum with the continuum-fitting mask shaded.
    """
    wave = np.asarray(wave)
    mask = np.asarray(mask, dtype=bool)

    ax = plot_spectrum(wave, flux, error=error, ax=ax, stage=stage, flux_unit=flux_unit)

    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for s, e in zip(edges[0::2], edges[1::2] - 1):
        ax.axvspan(wave[s], wave[e], color='C1', alpha=0.3)

    if save is not None:
        ax.figure.savefig(save, dpi=150, bbox_inches='tight')

    return ax    
