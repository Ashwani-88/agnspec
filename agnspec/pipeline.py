"""
``run_pipeline``: read a spectrum, deredden, shift to the rest frame, fit a
power-law continuum, subtract it, fit one emission-line complex, measure the
lines, estimate bootstrap errors, make plots and save everything.

This is the quick, single-complex route. For anything more involved (Fe II,
the Balmer pseudo-continuum, MCMC) build the model yourself as shown in the
example notebooks.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

from . import io as agn_io
from . import preprocessing as prep
from .continuum import PowerLaw, get_continuum_mask
from .emission_lines import extract_region
from .fitting import LeastSquares
from .models import AGNModel
from . import uncertainties as unc
from . import plotting as plots

# Default line-free continuum windows (rest frame, Angstrom). Adjust them to
# the wavelength coverage of your spectra.
DEFAULT_CONTINUUM_WINDOWS = [
    (4200, 4230), (5080, 5130), (6000, 6050), (7450, 7550),
]

# Default fitting window around each supported line complex (rest frame, Angstrom).
DEFAULT_LINE_WINDOWS = {
    "Hbeta": (4750, 5100),
    "Halpha": (6400, 6800),
    "MgII": (2700, 2900),
    "CIV": (1450, 1700),
    "OIII": (4950, 5050),
}


def run_pipeline(filename: str, z: float, ebv: float = 0.0, line: str = "Hbeta",
                  continuum_windows: Optional[Sequence[Tuple[float, float]]] = None,
                  line_window: Optional[Tuple[float, float]] = None,
                  bad_regions: Optional[Sequence[Tuple[float, float]]] = None,
                  fitter=None, n_boot: int = 200,
                  output_dir: Optional[str] = None, make_plots: bool = True):
    """
    Fit one emission-line complex in a single spectrum, start to finish.

    Parameters
    ----------
    filename : str
        Path to a FITS/ASCII/CSV spectrum (see :func:`io.read_spectrum`).
    z : float
        Redshift, used to shift the spectrum to the rest frame.
    ebv : float
        Galactic E(B-V) for foreground dereddening (default: 0, i.e.
        skip dereddening).
    line : str
        Which emission-line complex to fit: one of
        ``"Hbeta", "Halpha", "MgII", "CIV", "OIII"``.
    continuum_windows : list of (float, float), optional
        Rest-frame continuum windows; defaults to
        :data:`DEFAULT_CONTINUUM_WINDOWS`.
    line_window : (float, float), optional
        Rest-frame wavelength range to extract for the line fit;
        defaults to :data:`DEFAULT_LINE_WINDOWS[line]`.
    bad_regions : list of (float, float), optional
        Extra wavelength regions to mask (telluric bands, chip gaps).
    fitter : fitting.Fitter, optional
        Defaults to :class:`fitting.LeastSquares`.
    n_boot : int
        Number of bootstrap iterations for uncertainty estimation
        (set to 0 to skip).
    output_dir : str, optional
        If given, plots and a JSON results file are written here.
    make_plots : bool
        Whether to generate diagnostic plots.

    Returns
    -------
    model : models.AGNModel
        The fitted model object (continuum + line complex + results).
    results : dict
        Flat dict of all fitted parameters and derived measurements,
        the same dict written to disk if ``output_dir`` is given.
    """
    fitter = fitter or LeastSquares()
    continuum_windows = continuum_windows or DEFAULT_CONTINUUM_WINDOWS
    line_window = line_window or DEFAULT_LINE_WINDOWS.get(line, None)
    if line_window is None:
        raise ValueError(f"No default window for line '{line}'; please supply line_window=.")

    # ---- 1. Read -------------------------------------------------
    wave_obs, flux_obs, error_obs, header = agn_io.read_spectrum(filename)

    # ---- 2. Deredden (Galactic, observed frame) -------------------
    if ebv:
        if error_obs is not None:
            flux_dered, error_dered = prep.deredden_spectrum(
                wave_obs, flux_obs, error=error_obs, ebv=ebv)
        else:
            flux_dered, error_dered = prep.deredden_spectrum(wave_obs, flux_obs, ebv=ebv), None
    else:
        flux_dered, error_dered = flux_obs, error_obs

    # ---- 3. Shift to rest frame (wavelength ONLY; flux untouched) ----
    wave_rest = prep.to_rest_frame(wave_obs, z)
    flux_rest = flux_dered
    error_rest = error_dered

    # ---- 4. Mask bad pixels -----------------------------------------
    good = prep.mask_bad_pixels(wave_rest, flux_rest, error_rest, bad_regions=bad_regions)
    wave_rest, flux_rest = wave_rest[good], flux_rest[good]
    error_rest = error_rest[good] if error_rest is not None else None

    # ---- 5. Fit continuum --------------------------------------------
    model = AGNModel(continuum_components=[PowerLaw()], fitter=fitter)
    continuum_result = model.fit_continuum(
        wave_rest, flux_rest, error=error_rest, continuum_windows=continuum_windows
    )
    continuum_flux_full = model.continuum_flux(wave_rest)

    # ---- 6. Subtract continuum ---------------------------------------
    line_flux_full = flux_rest - continuum_flux_full

    # ---- 7. Fit emission line(s) --------------------------------------
    model.add_line_complex(line)
    xmin, xmax = line_window
    w_line, f_line = extract_region(wave_rest, line_flux_full, xmin, xmax)
    e_line = None
    if error_rest is not None:
        _, e_line = extract_region(wave_rest, error_rest, xmin, xmax)

    line_result = model.fit_line(line, w_line, f_line, e_line)

    # ---- 8. Measure line properties ------------------------------------
    line_measurements = model.measure_line(line)

    # ---- 9. Bootstrap uncertainties --
    boot_summary = {}
    if n_boot and n_boot > 0:
        complex_obj = model.line_complexes[line]
        try:
            _, param_std, _ = unc.bootstrap_errors(
                w_line, f_line, e_line, complex_obj, n_boot=n_boot, fitter=fitter,
            )
            boot_summary = dict(zip(complex_obj.free_param_names, param_std))
        except Exception as exc:
            boot_summary = {"bootstrap_error": str(exc)}

    # ---- 10. Generate plots ---------------------------------------------
    figures = {}
    if make_plots:
        ax1 = plots.plot_spectrum(wave_rest, flux_rest)
        figures["spectrum"] = ax1.figure

        fig2 = model.plot_continuum()
        figures["continuum"] = fig2.figure if hasattr(fig2, "figure") else fig2

        fig3, _ = model.plot_line(line, w_line, f_line,
                                   continuum=None)
        figures["line_fit"] = fig3

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150,
                            bbox_inches="tight")

    # ---- 11. Save outputs -------------------------------------------------
    results = {
        "filename": filename,
        "redshift": z,
        "ebv": ebv,
        "line": line,
        "continuum": continuum_result.as_dict(),
        "line_fit": line_result.as_dict(),
        "line_measurements": line_measurements,
        "bootstrap_param_errors": boot_summary,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filename))[0]
        agn_io.save_results(os.path.join(output_dir, f"{base}_results.json"), results)

    return model, results
