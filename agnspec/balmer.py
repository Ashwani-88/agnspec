"""
balmer.py
=========

Balmer *pseudo*-continuum following Kovacevic, Popovic & Kollatschny
(2013, Adv. Space Res.), "A model for the Balmer pseudocontinuum in
spectra of type 1 AGNs" (hereafter KPK13).

Physics / recipe (paper Sect. 2)
--------------------------------
* lambda > lambda_BE : a *series* of broad Balmer lines (upper level
  n = 4 ... 400 -- H-beta, H-gamma, ... n=400), all with the same
  velocity width and the same velocity shift, with *fixed relative
  intensities* (Storey & Hummer 1995, case B, T=15000 K, ne=1e10 for
  n <= 50; Boltzmann factor, paper Eq. 4, for 50 < n <= 400).
  Their overlap produces the smooth rise to the Balmer edge.
* lambda <= lambda_BE : partially optically-thick Balmer continuum
  (Grandi 1982), uniform T_e = 15000 K and tau_BE = 1 (Kurk+2007):

      F(l) = F_BaC * B_l(Te) * (1 - exp(-tau_l)),  tau_l = tau_BE (l_BE/l)^3

  where F_BaC is *not* free: requiring continuity at the edge,
  F_BaC * B_BE(Te) (1 - exp(-tau_BE)) = sum_{i>=6} G_i(lambda_BE)
  (paper Eqs. 5 and 7).

Free parameters of the component: integrated flux, velocity width and
velocity shift of the H-beta Gaussian of the series.  (With the
power-law component, KPK13's "four free parameters" are alpha, W,
delta and I(H-beta); the power-law normalisation F_5600 is a fifth.)

Implementation notes / deliberate choices (see README_BALMER_CHANGES.md)
-----------------------------------------------------------------------
* Wavelengths are *vacuum* (SDSS convention): lambda_n from the
  Rydberg formula with R_H = 109677.583 cm^-1, Balmer limit 3647.05 A.
* Each line is a flux-normalised Gaussian with sigma_n = lambda_n *
  sigma_v / c (common *velocity* width).  Relative intensities are
  treated as relative integrated *fluxes* (energy emissivities).
* The Storey & Hummer (1995) table is NOT bundled.  A built-in
  approximation (case-B decrement for n<=11 from the standard
  textbook values, ~ n^-3 beyond) is used unless you pass a real
  table (``table=`` or ``source="pyneb"``).  The result for the
  Balmer-edge height depends on these ratios, so use the real table
  for publication-grade numbers.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.integrate import simpson

from .continuum import ContinuumComponent, BalmerContinuum

C_KMS = 2.99792458e5
R_H = 109677.583            # cm^-1, hydrogen (reduced-mass) Rydberg
EDGE_VAC = 4.0e8 / R_H      # Angstrom (vacuum Balmer limit, 3647.05)
EV_ZERO_TO_IONIZATION = 13.598434   # eV
K_B_EV = 8.617333262e-5     # eV / K


def balmer_wavelengths(n):
    """Vacuum wavelengths (A) of H(n -> 2) lines, n >= 3."""
    n = np.asarray(n, dtype=float)
    return 1.0e8 / (R_H * (0.25 - 1.0 / n ** 2))


# ----------------------------------------------------------------------
# Relative line intensities
# ----------------------------------------------------------------------
# Case-B Balmer decrement relative to H-beta (=1) at T ~ 1e4 K,
# ne ~ 1e2-1e4 cm^-3 (textbook values, e.g. Osterbrock & Ferland 2006,
# Table 4.4).  APPROXIMATE stand-in for Storey & Hummer (1995).
_BUILTIN_RATIOS = {
    4: 1.000,    # H-beta
    5: 0.468,    # H-gamma
    6: 0.259,    # H-delta
    7: 0.159,
    8: 0.105,
    9: 0.0731,
    10: 0.0530,
    11: 0.0398,
}
_N3_COEFF = 52.5   # I_n / I_Hbeta ~ 52.5 / n^3 for n >= 12 (fitted to n = 8-11)


def builtin_ratio_table(n_max=50):
    """{n: I_n/I_Hbeta} for n = 4..n_max from the built-in approximation."""
    tab = dict(_BUILTIN_RATIOS)
    for n in range(12, n_max + 1):
        tab[n] = _N3_COEFF / n ** 3
    return tab


def pyneb_ratio_table(T=15000.0, ne=1.0e10, n_max=50):
    """
    Case-B H I emissivity ratios relative to H-beta from PyNeb's
    Storey & Hummer (1995) tables (optional dependency).  Levels that
    PyNeb cannot provide are extrapolated with the n^-3 law from the
    highest level available.
    """
    import pyneb as pn
    atom = pn.RecAtom("H", 1)
    ref = float(atom.getEmissivity(T, ne, 4, 2))
    tab = {}
    last_n = 4
    for n in range(4, n_max + 1):
        try:
            val = float(atom.getEmissivity(T, ne, n, 2))
            if not np.isfinite(val) or val <= 0:
                raise ValueError
            tab[n] = val / ref
            last_n = n
        except Exception:
            tab[n] = tab[last_n] * (last_n / n) ** 3
    return tab


def load_ratio_table(path):
    """
    Load a two-column text file ``n  I_n/I_Hbeta`` (n = 4 ... N).
    Use this to supply the real Storey & Hummer (1995) numbers for
    T=15000 K, ne=1e10 cm^-3, case B.
    """
    arr = np.loadtxt(path)
    return {int(n): float(r) for n, r in arr}


def balmer_ratios(n, T=15000.0, high_n="boltzmann", n_table_max=50, table=None):
    """
    Relative intensities I_n / I_Hbeta for upper levels ``n`` (>= 4).

    n <= n_table_max : lookup in ``table`` (default: built-in approx.)
    n  > n_table_max : ``high_n="boltzmann"`` -> KPK13 Eq. (4):
                           I_n = I_50 * exp(-(E_n - E_50)/kT)
                       ``high_n="n3"``        -> I_n = I_50 (50/n)^3
                       (the asymptotic low-density recombination
                       scaling; provided for comparison/sensitivity).
    """
    n = np.asarray(n, dtype=int)
    if table is None:
        table = builtin_ratio_table(n_table_max)
    n_ref = min(n_table_max, max(table))
    i_ref = table[n_ref]
    out = np.empty(n.shape, dtype=float)
    lo = n <= n_ref
    out[lo] = [table[int(k)] for k in n[lo]]
    hi = ~lo
    if np.any(hi):
        nh = n[hi].astype(float)
        if high_n == "boltzmann":
            e_n = EV_ZERO_TO_IONIZATION * (1.0 - 1.0 / nh ** 2)
            e_ref = EV_ZERO_TO_IONIZATION * (1.0 - 1.0 / n_ref ** 2)
            out[hi] = i_ref * np.exp(-(e_n - e_ref) / (K_B_EV * T))
        elif high_n == "n3":
            out[hi] = i_ref * (n_ref / nh) ** 3
        else:
            raise ValueError("high_n must be 'boltzmann' or 'n3'")
    return out


# ----------------------------------------------------------------------
# The component
# ----------------------------------------------------------------------
class BalmerPseudoContinuum(ContinuumComponent):
    """
    KPK13 Balmer pseudo-continuum (high-order Balmer lines for
    lambda > edge, partially thick Balmer continuum for lambda <= edge).

    Parameters (of ``model``)
    -------------------------
    flux_hb  : integrated flux of the H-beta Gaussian of the series
               (flux-density units x Angstrom).
    sigma_v  : Gaussian velocity width of every line (km/s, sigma).
    dv       : common velocity shift (km/s).

    Constructor options
    -------------------
    te, tau_be : electron temperature (K) and optical depth at the edge
                 (KPK13: 15000 K, 1.0).
    n_max      : highest upper level (400).
    n_edge_min : first level entering the edge sum, paper Eq. (5) -> 6.
    high_n     : "boltzmann" (KPK13 Eq. 4) or "n3".
    table      : optional {n: ratio} (e.g. real Storey & Hummer values).
    """

    name = "balmer_pseudo"
    param_names = ["flux_hb", "sigma_v", "dv"]

    def __init__(self, te=15000.0, tau_be=1.0, n_max=400, n_edge_min=6,
                 high_n="boltzmann", table=None, n_table_max=50):
        self.te = float(te)
        self.tau_be = float(tau_be)
        self.n_max = int(n_max)
        self.n_edge_min = int(n_edge_min)
        self.high_n = high_n
        self.edge = EDGE_VAC

        self.n = np.arange(4, self.n_max + 1)
        self.lam_n = balmer_wavelengths(self.n)
        self.ratio_n = balmer_ratios(self.n, T=self.te, high_n=high_n,
                                     n_table_max=n_table_max, table=table)
        self._edge_sel = self.n >= self.n_edge_min

    # ---- pieces ------------------------------------------------------
    def _line_sum(self, wave, flux_hb, sigma_v, dv, sel=None):
        """Sum of Gaussians at ``wave`` (all or a subset ``sel`` of levels)."""
        wave = np.asarray(wave, dtype=float)
        lam = self.lam_n * (1.0 + dv / C_KMS)
        sig = lam * sigma_v / C_KMS
        flx = flux_hb * self.ratio_n
        if sel is not None:
            lam, sig, flx = lam[sel], sig[sel], flx[sel]
        z = (wave[:, None] - lam[None, :]) / sig[None, :]
        return np.sum(flx / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-0.5 * z * z), axis=1)

    def edge_flux_density(self, flux_hb, sigma_v, dv):
        """sum_{i>=n_edge_min} G_i(lambda_BE)   (paper Eq. 5, right side)."""
        return float(self._line_sum([self.edge], flux_hb, sigma_v, dv,
                                    sel=self._edge_sel)[0])

    def f_bac(self, flux_hb, sigma_v, dv):
        """Paper Eq. (7): Balmer-continuum scale F_BaC (not a free parameter)."""
        b_edge = BalmerContinuum._planck_lambda(self.edge, self.te)
        return self.edge_flux_density(flux_hb, sigma_v, dv) / (
            b_edge * (-np.expm1(-self.tau_be)))

    def balmer_continuum(self, wave, flux_hb, sigma_v, dv):
        """Balmer continuum (zero for lambda > edge)."""
        wave = np.asarray(wave, dtype=float)
        out = np.zeros_like(wave)
        below = wave <= self.edge
        if np.any(below):
            w = wave[below]
            tau = self.tau_be * (self.edge / w) ** 3
            out[below] = self.f_bac(flux_hb, sigma_v, dv) * \
                BalmerContinuum._planck_lambda(w, self.te) * (-np.expm1(-tau))
        return out

    def balmer_lines(self, wave, flux_hb, sigma_v, dv):
        """High-order Balmer-line series (zero for lambda <= edge, as in KPK13)."""
        wave = np.asarray(wave, dtype=float)
        out = np.zeros_like(wave)
        above = wave > self.edge
        if np.any(above):
            out[above] = self._line_sum(wave[above], flux_hb, sigma_v, dv)
        return out

    # ---- ContinuumComponent interface --------------------------------
    def model(self, wave, flux_hb, sigma_v, dv):
        return (self.balmer_continuum(wave, flux_hb, sigma_v, dv)
                + self.balmer_lines(wave, flux_hb, sigma_v, dv))

    def initial_guess(self, wave, flux):
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        sel = (wave > 4750) & (wave < 4980)
        if sel.sum() > 3:
            base = np.nanmedian(flux[(wave > 5080) & (wave < 5130)]) \
                if np.any((wave > 5080) & (wave < 5130)) else np.nanmedian(flux[sel])
            area = max(float(simpson(np.clip(flux[sel] - base, 0, None), x=wave[sel])), 1.0)
        else:
            area = 100.0
        return [area, 2500.0, 0.0]

    def bounds(self):
        return ([0.0, 300.0, -3000.0], [np.inf, 10000.0, 3000.0])

    # ---- derived quantities ------------------------------------------
    def integrated_balmer_continuum(self, flux_hb, sigma_v, dv, lam_lo=912.0, n=4000):
        """Integral of the Balmer continuum over [lam_lo, edge] (flux x A)."""
        w = np.linspace(lam_lo, self.edge, n)
        return float(simpson(self.balmer_continuum(w, flux_hb, sigma_v, dv), x=w))

    def components(self, wave, params) -> Dict[str, np.ndarray]:
        flux_hb, sigma_v, dv = params
        return {
            "balmer_continuum": self.balmer_continuum(wave, flux_hb, sigma_v, dv),
            "balmer_lines": self.balmer_lines(wave, flux_hb, sigma_v, dv),
        }
