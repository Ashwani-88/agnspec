"""
Generate a synthetic AGN-like spectrum (power law + Fe II-free Hbeta
region with broad+narrow Hbeta and the [OIII] doublet) and run it
through the full agnspec pipeline as a smoke test / usage example.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agnspec.utils import gaussian, LINE_WAVELENGTHS
from agnspec.continuum import PowerLaw
from agnspec.emission_lines import HbetaComplex
from agnspec.fitting import LeastSquares
from agnspec.models import AGNModel
from agnspec import preprocessing as prep
from agnspec import measurements as meas


def make_synthetic_spectrum(seed=42):
    rng = np.random.default_rng(seed)
    wave = np.arange(4600, 5150, 1.0)

    pl = PowerLaw(wave0=5100.0)
    continuum = pl.model(wave, amplitude=5.0, alpha=-1.6)

    hb_broad = gaussian(wave, 8.0, LINE_WAVELENGTHS["Hbeta"], 18.0)
    hb_narrow = gaussian(wave, 2.0, LINE_WAVELENGTHS["Hbeta"], 3.5)
    oiii_5007 = gaussian(wave, 6.0, LINE_WAVELENGTHS["OIII_5007"], 3.0)
    oiii_4959 = gaussian(wave, 6.0 / 2.98, LINE_WAVELENGTHS["OIII_4959"], 3.0)

    flux_clean = continuum + hb_broad + hb_narrow + oiii_5007 + oiii_4959
    noise_sigma = 0.05 * np.nanmax(flux_clean)
    flux = flux_clean + rng.normal(scale=noise_sigma, size=wave.size)
    error = np.full_like(wave, noise_sigma)

    return wave, flux, error, {
        "hb_broad": (8.0, LINE_WAVELENGTHS["Hbeta"], 18.0),
        "hb_narrow": (2.0, LINE_WAVELENGTHS["Hbeta"], 3.5),
        "oiii_5007": (6.0, LINE_WAVELENGTHS["OIII_5007"], 3.0),
    }


def main():
    wave, flux, error, truth = make_synthetic_spectrum()

    # Preprocessing steps (identity here since data are already
    # "rest frame" / dereddened, but exercised for completeness).
    wave_rest, flux_rest = prep.to_rest_frame(wave, flux, z=0.0)
    good = prep.mask_bad_pixels(wave_rest, flux_rest, error)
    wave_rest, flux_rest, error = wave_rest[good], flux_rest[good], error[good]

    model = AGNModel(continuum_components=[PowerLaw(wave0=5100.0)],
                      fitter=LeastSquares())

    windows = [(4700, 4750), (5080, 5130)]
    cont_result = model.fit_continuum(wave_rest, flux_rest, error=error,
                                       continuum_windows=windows)
    print("Continuum fit:", cont_result.as_dict())

    cont_flux = model.continuum_flux(wave_rest)
    line_flux = flux_rest - cont_flux

    model.add_line_complex("Hbeta", HbetaComplex(include_oiii=True))
    line_result = model.fit_line("Hbeta", wave_rest, line_flux, error)
    print("Line fit chi2/dof:", line_result.chi2, "/", line_result.dof)

    measurements = model.measure_line("Hbeta")
    for name, vals in measurements.items():
        print(name, {k: round(v, 3) for k, v in vals.items()})

    print("\nTruth (amp, center, sigma):")
    for k, v in truth.items():
        print(" ", k, v)

    ew = meas.equivalent_width(wave_rest, flux_rest, cont_flux)
    print("\nTotal Hbeta-region EW (approx):", round(ew, 2), "Angstrom")

    fig = model.plot_continuum()
    fig.figure.savefig("/tmp/agnspec_continuum_test.png", dpi=120, bbox_inches="tight")

    fig2, _ = model.plot_line("Hbeta", wave_rest, line_flux)
    fig2.savefig("/tmp/agnspec_line_test.png", dpi=120, bbox_inches="tight")

    print("\nSaved diagnostic plots to /tmp/agnspec_*_test.png")


if __name__ == "__main__":
    main()
