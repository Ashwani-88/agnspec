"""
io.py
=====

Reading spectra from disk (FITS / ASCII / CSV) and writing fit results
back out to disk (JSON).

FITS reading supports two distinct SDSS spectrum formats plus a
generic fallback:

  * **DR8 and later** (SDSS-III/BOSS, SDSS-IV/eBOSS, SDSS-V; filenames
    like ``spec-PLATE-MJD-FIBERID.fits``): the spectrum lives in a
    binary table (HDU ``COADD``, usually HDU 1) with ``flux``,
    ``loglam``, and ``ivar`` columns.
  * **DR7 and earlier classic** (filenames like
    ``spSpec-MJD-PLATE-FIBER.fit``): the spectrum lives in the
    *primary* HDU as a 2-D image, 4 rows x Npix, where row 0 is the
    flux, row 1 is the continuum-subtracted flux, row 2 is the noise
    (1-sigma error, same units as flux), and row 3 is a bitmask. The
    wavelength solution is log-linear, reconstructed from the
    ``COEFF0``/``COEFF1`` header keywords (or, in some files,
    ``CRVAL1``/``CD1_1``): lambda_i = 10**(COEFF0 + COEFF1 * i).
  * Any other simple 1-D image FITS spectrum with a linear WCS
    (``CRVAL1``/``CDELT1``/``CRPIX1``) is read as a generic fallback.

ASCII/CSV reading is unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Dict, Any

import numpy as np


def read_spectrum(filename: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Read a spectrum from FITS, ASCII, or CSV.

    For FITS files this transparently handles SDSS spectra from any
    data release (DR8+ BOSS-style binary tables and DR7-and-earlier
    classic spSpec image format), plus a generic linear-WCS FITS image
    fallback for non-SDSS files.

    Parameters
    ----------
    filename : str
        Path to the spectrum file.

    Returns
    -------
    wave : ndarray
        Wavelength (Angstrom).
    flux : ndarray
    error : ndarray or None
        1-sigma flux error, if available.
    header : dict
        Full FITS/ASCII header/metadata.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".fits", ".fit"):
        return _read_fits(filename)
    elif ext in (".csv",):
        return _read_ascii(filename, delimiter=",")
    else:
        return _read_ascii(filename)


def _read_fits(filename: str):
    from astropy.io import fits

    with fits.open(filename) as hdul:
        header = dict(hdul[0].header)

        # --- 1) DR8+ (BOSS/eBOSS/SDSS-V): flux/loglam/ivar binary table ---
        result = _read_sdss_dr8plus(hdul, header)
        if result is not None:
            return result

        # --- 2) DR7 and earlier: classic spSpec image format -------------
        result = _read_sdss_dr7(hdul, header)
        if result is not None:
            return result

        # --- 3) Generic fallback: simple 1-D image with linear WCS -------
        return _read_generic_fits_image(hdul, header, filename)


def _read_sdss_dr8plus(hdul, header):
    """
    SDSS DR8+ (BOSS/eBOSS/SDSS-V) format: a binary table extension
    (named ``COADD`` in modern files, but matched generically here)
    with ``flux``, ``loglam``, and ``ivar`` columns.

    Returns ``None`` if this HDU list doesn't look like this format,
    so the caller can fall through to the next reader.
    """
    table_hdu = None
    for hdu in hdul:
        if hdu.data is not None and hdu.is_image is False:
            colnames = [c.lower() for c in hdu.columns.names]
            if "flux" in colnames and "loglam" in colnames:
                table_hdu = hdu
                break

    if table_hdu is None:
        return None

    data = table_hdu.data
    colnames = [c.lower() for c in data.columns.names]

    def _get(*names):
        for n in names:
            if n in colnames:
                return np.array(data[data.columns.names[colnames.index(n)]], dtype=float)
        return None

    flux = _get("flux")
    wave = 10 ** np.array(data["loglam"], dtype=float)
    error = _get("ivar")

    if error is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            error = np.where(error > 0, 1.0 / np.sqrt(error), np.nan)

    return wave, flux, error, header


def _read_sdss_dr7(hdul, header):
    """
    Classic SDSS spectrum format (DR7 and earlier ``spSpec`` files):
    the primary HDU is a 2-D image, 4 (occasionally 5) rows x Npix:

        row 0 : flux
        row 1 : continuum-subtracted flux
        row 2 : noise (1-sigma error, same units as flux)
        row 3 : bitmask

    with a log-linear wavelength solution given by the ``COEFF0``/
    ``COEFF1`` header keywords (falling back to ``CRVAL1``/``CD1_1``
    if those aren't present): lambda_i = 10**(COEFF0 + COEFF1 * i).

    Returns ``None`` if this HDU list doesn't look like this format.
    """
    data = hdul[0].data
    if data is None or np.ndim(data) != 2:
        return None

    nrows, npix = data.shape
    if nrows not in (4, 5):
        return None

    coeff0 = header.get("COEFF0", header.get("CRVAL1"))
    coeff1 = header.get("COEFF1", header.get("CD1_1"))
    if coeff0 is None or coeff1 is None:
        return None

    flux = np.asarray(data[0], dtype=float)
    error = np.asarray(data[2], dtype=float)
    pix = np.arange(npix)
    wave = 10 ** (coeff0 + coeff1 * pix)

    return wave, flux, error, header


def _read_generic_fits_image(hdul, header, filename):
    """
    Generic fallback: a simple 1-D image spectrum with a linear WCS
    (``CRVAL1``/``CDELT1``/``CRPIX1``), for non-SDSS FITS spectra.
    """
    data = hdul[0].data
    if data is None:
        raise ValueError(f"No data found in FITS file {filename}")
    flux = np.asarray(data, dtype=float).flatten()

    crval1 = header.get("CRVAL1", 0.0)
    cdelt1 = header.get("CDELT1", header.get("CD1_1", 1.0))
    crpix1 = header.get("CRPIX1", 1.0)
    pix = np.arange(1, len(flux) + 1)
    wave = crval1 + (pix - crpix1) * cdelt1

    error = None
    if len(hdul) > 1 and hdul[1].data is not None:
        try:
            error = np.asarray(hdul[1].data, dtype=float).flatten()
        except Exception:
            error = None

    return wave, flux, error, header


def _read_ascii(filename: str, delimiter=None):
    data = np.genfromtxt(filename, delimiter=delimiter, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]
    wave = data[:, 0]
    flux = data[:, 1]
    error = data[:, 2] if ncols >= 3 else None

    header = {"source_file": filename, "n_columns": ncols}
    return wave, flux, error, header


def save_results(filename: str, parameters: Dict[str, Any]) -> None:
    """
    Save fit results (parameters, uncertainties, metadata) to a JSON
    file. Numpy arrays/scalars are converted to native Python types.

    Parameters
    ----------
    filename : str
        Output path (``.json`` recommended).
    parameters : dict
        Dictionary of results to serialize.
    """

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, (set,)):
            return list(obj)
        return str(obj)

    with open(filename, "w") as f:
        json.dump(parameters, f, indent=2, default=_default)


def load_results(filename: str) -> Dict[str, Any]:
    """Load a JSON results file previously written by :func:`save_results`."""
    with open(filename, "r") as f:
        return json.load(f)
