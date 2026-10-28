"""
Plots for each step of a fit: the spectrum at any preprocessing stage, the
continuum windows, continuum and line fits, and the Balmer pseudo-continuum fit.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# Used for the y-axis label when no unit is passed. SDSS spectra are usually
# scaled by 1e-17, so pass flux_unit=... (or flux_unit_from_header(header)) to
# get the right label for your data.
DEFAULT_FLUX_UNIT = r"erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$"


def flux_unit_from_header(header, default=DEFAULT_FLUX_UNIT):
    """
    Build a y-axis unit string from a FITS header. Looks for ``BUNIT``,
    ``FLUXUNIT`` or ``TUNIT2`` and returns the raw string, or ``default`` if
    none is present. Header unit strings are not LaTeX-formatted; pass your
    own ``flux_unit=`` to a plotting function if you want that.
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


# Label, colour and x-axis label for each stage passed to plot_spectrum.
_STAGE_INFO = {
    "raw": {
        "xlabel": r"Observed-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Raw spectrum",
        "color": "0.15",
    },
    "dered": {
        "xlabel": r"Observed-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Extinction-corrected spectrum",
        "color": "steelblue",
    },
    "smooth": {
        "xlabel": r"Rest-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Smoothed spectrum",
        "color": "crimson",
    },
    "rest": {
        "xlabel": r"Rest-frame Wavelength [$\mathrm{\AA}$]",
        "label": "Rest-frame spectrum",
        "color": "black",
    },
}


def plot_spectrum(wave, flux, error=None, ax=None, stage="raw",
                  label=None, color=None, flux_unit=None,
                  save=None, dpi=150, **kwargs):
    """
    Plot a spectrum at one stage of preprocessing. Call it several times with
    the same ``ax`` to overlay stages.

    Parameters
    ----------
    wave, flux : ndarray
    error : ndarray, optional
        Drawn as a shaded 1-sigma band.
    ax : matplotlib.axes.Axes, optional
    stage : {"raw", "dered", "smooth", "rest"}
        Sets the default label and colour, and whether the x-axis is labelled
        observed-frame ("raw", "dered") or rest-frame ("smooth", "rest").
    label, color : str, optional
        Override the defaults for the stage.
    flux_unit : str, optional
        Unit shown in the y-axis label, e.g.
        ``r"10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\\mathrm{\\AA}^{-1}$"``.
    save : str, optional
        If given, the figure is also written to this path.
    **kwargs
        Passed to ``ax.plot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if stage not in _STAGE_INFO:
        raise ValueError(f"Unknown stage '{stage}'. Use one of {list(_STAGE_INFO)}.")
    info = _STAGE_INFO[stage]
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


def plot_continuum_fit(wave, flux, model, components: Optional[Dict[str, np.ndarray]] = None,
                       ax=None, mask=None, flux_unit=None):
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
    ax.set_ylabel(_ylabel(flux_unit))
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
    """Plot the spectrum and shade the pixels selected by the continuum mask."""
    wave = np.asarray(wave)
    mask = np.asarray(mask, dtype=bool)

    ax = plot_spectrum(wave, flux, error=error, ax=ax, stage=stage, flux_unit=flux_unit)

    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for s, e in zip(edges[0::2], edges[1::2] - 1):
        ax.axvspan(wave[s], wave[e], color='C1', alpha=0.3)

    if save is not None:
        ax.figure.savefig(save, dpi=150, bbox_inches='tight')

    return ax


def plot_balmer_fit(wave, flux, powerlaw, balmer_lines, balmer_continuum,
                    flux_clean=None, fit_mask=None, flux_unit=None,
                    title=None, save=None, dpi=150, ylim=None, ylim_resid=None):
    """
    Two-panel plot in the style of Fig. 2 of Kovacevic, Popovic & Kollatschny
    (2014).

    Top panel: data, the Balmer continuum plus high-order Balmer lines
    (dotted), the power law (dashed) and the total (solid).
    Bottom panel: data minus (Balmer continuum + power law). The high-order
    Balmer lines stay in this residual, and the model lines are drawn dotted
    on top of it.

    Parameters
    ----------
    wave, flux : ndarray
        Rest-frame wavelength and the (dereddened) spectrum.
    powerlaw, balmer_lines, balmer_continuum : ndarray
        Model components evaluated on ``wave``.
    flux_clean : ndarray, optional
        Spectrum with the narrow lines and Fe II removed, drawn over the data.
    fit_mask : ndarray of bool, optional
        Pixels that were used in the fit.
    """
    wave = np.asarray(wave, dtype=float)
    pseudo = np.asarray(balmer_lines) + np.asarray(balmer_continuum)
    total = pseudo + np.asarray(powerlaw)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.08})

    ax1.plot(wave, flux, color="0.55", lw=0.7, label="data")
    if flux_clean is not None:
        ax1.plot(wave, flux_clean, color="k", lw=0.7, label="data (narrow lines + Fe II removed)")
    if fit_mask is not None:
        ax1.plot(wave[fit_mask], (flux_clean if flux_clean is not None else flux)[fit_mask],
                 ".", ms=2, color="dodgerblue", label="pixels used in fit", zorder=5)
    ax1.plot(wave, pseudo, ls=":", color="C3", lw=1.6,
             label="Balmer cont. + high-order Balmer lines")
    ax1.plot(wave, powerlaw, ls="--", color="C0", lw=1.4, label="power law")
    ax1.plot(wave, total, color="C3", lw=1.8, label="sum")
    ax1.set_ylabel(_ylabel(flux_unit))
    if ylim:
        ax1.set_ylim(*ylim)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    if title:
        ax1.set_title(title, fontsize=10)

    bc_pl = np.asarray(balmer_continuum) + np.asarray(powerlaw)
    ax2.plot(wave, np.asarray(flux) - bc_pl, color="k", lw=0.7,
             label="data - (Balmer cont. + power law)")
    ax2.plot(wave, np.asarray(balmer_lines), color="C3", lw=1.0, ls=":",
             label="high-order Balmer lines (model)")
    ax2.axhline(0, color="0.5", lw=0.6)
    ax2.set_xlabel(r"Rest-frame Wavelength [$\mathrm{\AA}$]")
    ax2.set_ylabel(_ylabel(flux_unit))
    if ylim_resid:
        ax2.set_ylim(*ylim_resid)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig, (ax1, ax2)
