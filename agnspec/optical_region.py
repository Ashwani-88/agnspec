"""
First step of the KPK14 recipe: decompose the 4000-5500 A region of a type 1
AGN so that only the broad Balmer lines and the continuum are left, giving
"clean" broad profiles (KPK14 Sect. 2, Fig. 1).

Components fitted together
--------------------------
* a local power law (a nuisance term in this step)
* optical Fe II (empirical template, Gaussian-broadened and shifted)
* narrow lines whose width and shift are fixed from [OIII]: the [OIII]
  4959,5007 core (3:1), narrow H-beta, H-gamma and H-delta, and [OIII] 4363
* an optional [OIII] blue-wing component (4959/5007 tied)
* broad He II 4686
* broad H-beta, H-gamma and H-delta, each the sum of two Gaussians with a
  common profile, so the broad wings are not absorbed by Fe II. H-gamma and
  H-delta are scaled copies of H-beta.

Differences from the paper
--------------------------
KPK14 use a multi-Gaussian model together with the Kovacevic et al. (2010)
Fe II template. That template is not part of agnspec, so the Park et al.
(2022) empirical Fe II template in ``agnspec/data`` is used instead. The
cleaned broad profiles are therefore similar to the paper's but not identical.

The class follows the same interface as the other models, so it can be passed
straight to :class:`agnspec.fitting.LeastSquares` or :class:`agnspec.fitting.MCMC`.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from .utils import LINE_WAVELENGTHS, C_KMS
from .continuum import FeII
from .balmer import balmer_wavelengths

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_park2022_feii(path=None):
    """Read the Park et al. (2022) optical Fe II template; returns (wave [A], flux [1e-15 cgs])."""
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
    feii_wave, feii_flux : ndarray
        Fe II template.
    narrow_sigma_v, narrow_dv : common narrow-line kinematics (km/s)
        (sigma and shift), normally measured from [OIII] 5007.
    pl_pivot : power-law pivot wavelength (A).
    include_wing : bool
        If True (default) a blue [OIII] wing component (4959/5007 tied) is
        fitted and removed.  If False the wing parameters, component and
        flux are dropped entirely from the model.
    """

    name = "optical_region"

    _WING_PARAMS = ("oiii_wing_flux", "oiii_wing_sigma", "oiii_wing_dv")

    def __init__(self, feii_wave, feii_flux, narrow_sigma_v, narrow_dv,
                 pl_pivot=5000.0, include_wing=True):
        self.feii = FeII(feii_wave, feii_flux)
        self.narrow_sigma_v = float(narrow_sigma_v)
        self.narrow_dv = float(narrow_dv)
        self.pl_pivot = pl_pivot
        self.include_wing = include_wing

        all_names = [
            "pl_amp", "pl_alpha",
            "feii_amp", "feii_sigma", "feii_dv",
            "oiii_flux", "oiii_wing_flux", "oiii_wing_sigma", "oiii_wing_dv",
            "hbn_flux", "o4363_flux",
            "heii_flux", "heii_sigma", "heii_dv",
            "bb1_flux", "bb1_sigma", "bb1_dv",
            "bb2_flux", "bb2_sigma", "bb2_dv",
            "hg_scale", "hd_scale",
        ]
        # one mask drives names, initial_guess and bounds, so they stay in
        # step when the wing is switched off
        self._keep = [include_wing or n not in self._WING_PARAMS for n in all_names]
        self.free_param_names = [n for n, k in zip(all_names, self._keep) if k]
        self.param_names = self.free_param_names

    def _narrow(self, wave, flux, rest):
        c = rest * (1 + self.narrow_dv / C_KMS)
        s = c * self.narrow_sigma_v / C_KMS
        return _g_flux(wave, flux, c, s)

    def components(self, wave, p) -> Dict[str, np.ndarray]:
        wave = np.asarray(wave, dtype=float)
        d = dict(zip(self.free_param_names, p))

        pl = d["pl_amp"] * (wave / self.pl_pivot) ** d["pl_alpha"]
        feii = self.feii.model(wave, d["feii_amp"], d["feii_sigma"], d["feii_dv"])

        # [OIII] core (4959 = 5007 / 2.98)
        oiii_core = (self._narrow(wave, d["oiii_flux"], L_O5007)
                     + self._narrow(wave, d["oiii_flux"] / 2.98, L_O4959))

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

        out = {"powerlaw": pl, "feii": feii, "oiii_core": oiii_core}

        # [OIII] blue-wing component (4959/5007 tied), only if requested
        if self.include_wing:
            wsig = d["oiii_wing_sigma"]
            oiii_wing = 0.0
            for rest, fl in ((L_O5007, d["oiii_wing_flux"]),
                             (L_O4959, d["oiii_wing_flux"] / 2.98)):
                c = rest * (1 + (self.narrow_dv + d["oiii_wing_dv"]) / C_KMS)
                oiii_wing = oiii_wing + _g_flux(wave, fl, c, c * wsig / C_KMS)
            out["oiii_wing"] = oiii_wing

        out.update({
            "narrow_balmer": narrow_balmer,
            "oiii_4363": o4363,
            "heii_4686": heii,
            "broad_hbeta": broad(L_HB, 1.0),
            "broad_hgamma": broad(L_HG, d["hg_scale"]),
            "broad_hdelta": broad(L_HD, d["hd_scale"]),
        })
        return out

    def model(self, wave, *params):
        return sum(self.components(wave, params).values())

    # components that are subtracted to leave the "clean" broad Balmer lines
    REMOVED = ("feii", "oiii_core", "oiii_wing", "narrow_balmer", "oiii_4363", "heii_4686")

    def removable(self, wave, params):
        comps = self.components(wave, params)
        return sum(comps[k] for k in self.REMOVED if k in comps)

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
        guess = [
            cont, -1.0,                         # power law
            0.5 * cont, 1200.0, 0.0,            # Fe II
            peak_o * sn * 2.5, 0.2 * peak_o * sn * 2.5, 600.0, -300.0,   # [OIII]
            0.05 * area_hb, 1.0,                # narrow Hb, [OIII]4363
            1.0, 2000.0, 0.0,                   # He II
            0.6 * area_hb, 1800.0, 0.0,         # broad 1
            0.4 * area_hb, 4000.0, 0.0,         # broad 2
            0.4, 0.25,                          # Hg/Hb, Hd/Hb broad
        ]
        return [g for g, k in zip(guess, self._keep) if k]

    def bounds(self):
        inf = np.inf
        lo = [0, -4,  0,  300, -1500,   0, 0, 250, -1500,   0, 0,   0, 1000, -1500,
              0,  800, -1500,   0, 2500, -2500, 0.05, 0.02]
        hi = [inf, 4, inf, 3000, 1500, inf, inf, 1500, 0, inf, inf, inf, 3000, 1500,
              inf, 3500, 1500, inf, 8000, 2500, 1.5, 1.0]
        return ([v for v, k in zip(lo, self._keep) if k],
                [v for v, k in zip(hi, self._keep) if k])

    def line_fluxes(self, params) -> Dict[str, float]:
        d = dict(zip(self.free_param_names, params))
        out = {
            "Hbeta_broad": d["bb1_flux"] + d["bb2_flux"],
            "Hbeta_narrow": d["hbn_flux"],
            "OIII5007_core": d["oiii_flux"],
            "HeII4686": d["heii_flux"],
        }
        if self.include_wing:
            out["OIII5007_wing"] = d["oiii_wing_flux"]
        return out
