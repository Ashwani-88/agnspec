"""
Constants and small helpers shared across agnspec.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import requests

IRSA_URL = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust"

# Physical constants (cgs unless noted)
C_KMS = 2.99792458e5          # speed of light, km/s
H_PLANCK = 6.62607015e-27     # erg s
K_BOLTZ = 1.380649e-16        # erg / K


def air_to_vac(wave_air):
    """
    Convert air wavelengths (Angstrom, valid above ~2000 A) to vacuum with
    the IAU standard formula of Morton (2000).
    """
    wave_air = np.asarray(wave_air, dtype=float)
    s = 1e4 / wave_air
    n = (1 + 0.00008336624212083 +
         0.02408926869968 / (130.1065924522 - s ** 2) +
         0.0001599740894897 / (38.92568793293 - s ** 2))
    return wave_air * n


# Rest-frame line wavelengths in vacuum (Angstrom).
# The optical lines below are the usual laboratory air wavelengths and are
# converted with air_to_vac, because SDSS spectra are tabulated in vacuum
# (this moves H-beta from 4861.33 to 4862.69 A, about 80 km/s). The UV lines
# and [OII] are already vacuum values.
_OPTICAL_AIR = {
    "Hdelta": 4101.73,
    "Hgamma": 4340.47,
    "Hbeta": 4861.33,
    "OIII_4959": 4958.91,
    "OIII_5007": 5006.84,
    "HeI_5876": 5875.62,
    "OI_6300": 6300.30,
    "Halpha": 6562.80,
    "NII_6548": 6548.05,
    "NII_6584": 6583.45,
    "SII_6716": 6716.44,
    "SII_6731": 6730.82,
}

LINE_WAVELENGTHS = {
    "Lyalpha": 1215.67,
    "CIV": 1549.06,
    "CIII]": 1908.73,
    "MgII": 2798.75,
    "OII": 3727.09,
}
LINE_WAVELENGTHS.update({k: float(air_to_vac(v)) for k, v in _OPTICAL_AIR.items()})
LINE_WAVELENGTHS = dict(sorted(LINE_WAVELENGTHS.items(), key=lambda kv: kv[1]))


def gaussian(x, amplitude, center, sigma):
    """Gaussian profile: amplitude * exp(-0.5 * ((x - center) / sigma)**2)."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def sigma_to_fwhm(sigma):
    """Convert a Gaussian sigma to FWHM."""
    return sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))


def get_ebv(ra, dec, reg_size=2.0, stat="meanValueSandF"):
    """
    Look up Galactic E(B-V) at one position from the IRSA DUST service.

    Parameters
    ----------
    ra, dec : float
        Coordinates in degrees (J2000).
    reg_size : float, default 2.0
        Size in degrees of the image cutout the service builds (allowed
        range 2.0 to 37.5). It does not change the returned statistic.
    stat : str, default "meanValueSandF"
        Which statistic to return. The IRSA tags are
        ``meanValueSandF`` (mean within 5 arcmin, Schlafly & Finkbeiner 2011),
        ``refPixelValueSandF`` (value at the central pixel) and
        ``stdSandF`` (scatter within 5 arcmin). Replace ``SandF`` with
        ``SFD`` to get the Schlegel, Finkbeiner & Davis (1998) values instead.

    Returns
    -------
    float
        E(B-V) in magnitudes, or NaN if the query fails.
    """
    if not (2.0 <= reg_size <= 37.5):
        print(f"reg_size={reg_size} is outside the allowed range 2.0-37.5")
        return np.nan

    params = {"locstr": f"{ra:.6f} {dec:.6f} equ J2000", "regSize": reg_size}

    try:
        r = requests.get(IRSA_URL, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)

        # the service can answer HTTP 200 and still report an error in the XML
        if root.get("status") != "ok":
            msg = (root.findtext("message") or "unknown error").strip()
            print(f"IRSA error for RA={ra}, DEC={dec}: {msg}")
            return np.nan

        value = root.findtext(f".//{stat}")
        if value is None:
            print(f"'{stat}' not found in the IRSA reply for RA={ra}, DEC={dec}")
            return np.nan

        return float(value.split()[0])      # "0.0301 (mag)" -> 0.0301

    except Exception as e:
        print(f"E(B-V) query failed for RA={ra}, DEC={dec}: {e}")
        return np.nan
