"""
optical_region.py
=================

Stage 1 of the KPK13 recipe: decompose the 4000-5500 A region of a
type-1 AGN so that *only the broad Balmer lines (plus continuum)*
remain once everything else is removed ("clean broad profiles",
paper Sect. 2, Fig. 1).

Components fitted simultaneously
--------------------------------
* local power law (nuisance in this stage)
* optical Fe II (empirical template, Gaussian-broadened and shifted)
* narrow lines with *common kinematics fixed from [OIII]* (paper:
  "we use the prominent narrow [OIII] lines to fix the width and the
  shift for all narrow components"): [OIII]4959,5007 core (3:1),
  narrow H-beta/gamma/delta, [OIII]4363
* [OIII] blue-wing component (4959/5007 tied)
* broad He II 4686
* broad H-beta / H-gamma / H-delta, each as the sum of two Gaussians
  with a common profile (flexible enough to keep the broad wings from
  being absorbed by Fe II); H-gamma / H-delta are scaled copies of H-beta.

Differences from the paper
--------------------------
KPK13 use the multi-Gaussian + Kovacevic et al. (2010) Fe II template
(an online model).  That template is not part of agnspec, so the Park
et al. (2022) empirical Fe II template shipped in ``agnspec/data`` is
used instead.  Results for the cleaned broad profiles are therefore
similar but not identical.

The class follows the agnspec model interface, so it can be passed to
``agnspec.fitting.LeastSquares`` (or MCMC) unchanged.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d

from .utils import LINE_WAVELENGTHS, C_KMS
from .continuum import FeII
from .balmer import balmer_wavelengths

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_park2022_feii(path=None):
    """Park et al. (2022) optical Fe II template -> (wave [A], flux [1e-15 cgs])."""
    path = path or os.path.join(DATA_DIR, "irontemplate_park2022.dat")
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 3:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def _g_flux(wave, flux, center, sigma):
    """Flux-normalised Gaussian (integrates to ``flux``)."""
    return flux / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((wave - center) / sigma) ** 2)


# Wavelengths (vacuum)
L_HB, L_HG, L_HD = (float(balmer_wavelengths(n)) for n in (4, 5, 6))
L_O5007 = LINE_WAVELENGTHS["OIII_5007"]
L_O4959 = LINE_WAVELENGTHS["OIII_4959"]
L_O4363 = 4364.436          # [OIII] 4363 (vacuum)
L_HEII = 4687.02            # He II 4686 (vacuum)
NARROW_HG_OVER_HB = 0.468   # case-B decrement for the narrow lines
NARROW_HD_OVER_HB = 0.259


class OpticalRegionModel:
    """
    Parameters
    ----------
    feii_wave, feii_flux : Fe II template.
    narrow_sigma_v, narrow_dv : common narrow-line kinematics (km/s)
        (sigma and shift), normally measured from [OIII] 5007.
    pl_pivot : power-law pivot wavelength (A).
    """

    name = "optical_region"

    def __init__(self, feii_wave, feii_flux, narrow_sigma_v, narrow_dv, pl_pivot=5000.0):
        self.feii = FeII(feii_wave, feii_flux)
        self.narrow_sigma_v = float(narrow_sigma_v)
        self.narrow_dv = float(narrow_dv)
        self.pl_pivot = pl_pivot
        self.free_param_names = [
            "pl_amp", "pl_alpha",
            "feii_amp", "feii_sigma", "feii_dv",
            "oiii_flux", "oiii_wing_flux", "oiii_wing_sigma", "oiii_wing_dv",
            "hbn_flux", "o4363_flux",
            "heii_flux", "heii_sigma", "heii_dv",
            "bb1_flux", "bb1_sigma", "bb1_dv",
            "bb2_flux", "bb2_sigma", "bb2_dv",
            "hg_scale", "hd_scale",
        ]
        self.param_names = self.free_param_names

    # ---- helpers -------------------------------------------------------
    def _narrow(self, wave, flux, rest):
        c = rest * (1 + self.narrow_dv / C_KMS)
        s = c * self.narrow_sigma_v / C_KMS
        return _g_flux(wave, flux, c, s)

    def components(self, wave, p) -> Dict[str, np.ndarray]:
        wave = np.asarray(wave, dtype=float)
        d = dict(zip(self.free_param_names, p))

        pl = d["pl_amp"] * (wave / self.pl_pivot) ** d["pl_alpha"]
        feii = self.feii.model(wave, d["feii_amp"], d["feii_sigma"], d["feii_dv"])

        # [OIII] core + wing (4959 = 5007 / 2.98)
        oiii_core = (self._narrow(wave, d["oiii_flux"], L_O5007)
                     + self._narrow(wave, d["oiii_flux"] / 2.98, L_O4959))
        wsig = d["oiii_wing_sigma"]
        oiii_wing = 0.0
        for rest, fl in ((L_O5007, d["oiii_wing_flux"]), (L_O4959, d["oiii_wing_flux"] / 2.98)):
            c = rest * (1 + (self.narrow_dv + d["oiii_wing_dv"]) / C_KMS)
            oiii_wing = oiii_wing + _g_flux(wave, fl, c, c * wsig / C_KMS)

        narrow_balmer = (self._narrow(wave, d["hbn_flux"], L_HB)
                         + self._narrow(wave, d["hbn_flux"] * NARROW_HG_OVER_HB, L_HG)
                         + self._narrow(wave, d["hbn_flux"] * NARROW_HD_OVER_HB, L_HD))
        o4363 = self._narrow(wave, d["o4363_flux"], L_O4363)

        c = L_HEII * (1 + d["heii_dv"] / C_KMS)
        heii = _g_flux(wave, d["heii_flux"], c, c * d["heii_sigma"] / C_KMS)

        def broad(rest, scale):
            out = 0.0
            for k in ("bb1", "bb2"):
                c = rest * (1 + d[f"{k}_dv"] / C_KMS)
                out = out + _g_flux(wave, scale * d[f"{k}_flux"], c, c * d[f"{k}_sigma"] / C_KMS)
            return out

        broad_hb = broad(L_HB, 1.0)
        broad_hg = broad(L_HG, d["hg_scale"])
        broad_hd = broad(L_HD, d["hd_scale"])

        return {
            "powerlaw": pl,
            "feii": feii,
            "oiii_core": oiii_core,
            "oiii_wing": oiii_wing,
            "narrow_balmer": narrow_balmer,
            "oiii_4363": o4363,
            "heii_4686": heii,
            "broad_hbeta": broad_hb,
            "broad_hgamma": broad_hg,
            "broad_hdelta": broad_hd,
        }

    def model(self, wave, *params):
        return sum(self.components(wave, params).values())

    # ---- the pieces that stage 1 REMOVES to give the "clean" spectrum --
    REMOVED = ("feii", "oiii_core", "oiii_wing", "narrow_balmer", "oiii_4363", "heii_4686")

    def removable(self, wave, params):
        comps = self.components(wave, params)
        return sum(comps[k] for k in self.REMOVED)

    # ---- fitter interface ----------------------------------------------
    def initial_guess(self, wave, flux):
        wave = np.asarray(wave)
        flux = np.asarray(flux)
        def med(lo, hi):
            m = (wave > lo) & (wave < hi)
            return float(np.nanmedian(flux[m])) if m.any() else float(np.nanmedian(flux))
        cont = med(5080, 5130)
        peak_hb = med(4800, 4900) - cont
        peak_o = max(float(np.nanmax(flux[(wave > 4990) & (wave < 5030)])) - cont, 1.0) \
            if ((wave > 4990) & (wave < 5030)).any() else 5.0
        sn = self.narrow_sigma_v / C_KMS * L_O5007
        area_hb = max(peak_hb, 5.0) * 40.0
        return [
            cont, -1.0,                         # power law
            0.5 * cont, 1200.0, 0.0,            # Fe II
            peak_o * sn * 2.5, 0.2 * peak_o * sn * 2.5, 600.0, -300.0,   # [OIII]
            0.05 * area_hb, 1.0,                # narrow Hb, [OIII]4363
            1.0, 2000.0, 0.0,                   # He II
            0.6 * area_hb, 1800.0, 0.0,         # broad 1
            0.4 * area_hb, 4000.0, 0.0,         # broad 2
            0.4, 0.25,                          # Hg/Hb, Hd/Hb broad
        ]

    def bounds(self):
        inf = np.inf
        lo = [0, -4,  0,  300, -1500,   0, 0, 250, -1500,   0, 0,   0, 1000, -1500,
              0,  800, -1500,   0, 2500, -2500, 0.05, 0.02]
        hi = [inf, 4, inf, 3000, 1500, inf, inf, 1500, 0, inf, inf, inf, 3000, 1500,
              inf, 3500, 1500, inf, 8000, 2500, 1.5, 1.0]
        return lo, hi

    # ---- fluxes ----------------------------------------------------------
    def line_fluxes(self, params) -> Dict[str, float]:
        d = dict(zip(self.free_param_names, params))
        return {
            "Hbeta_broad": d["bb1_flux"] + d["bb2_flux"],
            "Hbeta_narrow": d["hbn_flux"],
            "OIII5007_core": d["oiii_flux"],
            "OIII5007_wing": d["oiii_wing_flux"],
            "HeII4686": d["heii_flux"],
        }
