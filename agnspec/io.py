"""
Reading spectra (FITS, ASCII, CSV) and saving or loading fit results (JSON).

Three FITS layouts are understood:

* SDSS DR8 and later (``spec-PLATE-MJD-FIBERID.fits``): a binary table with
  ``flux``, ``loglam`` and ``ivar`` columns.
* SDSS DR7 and earlier (``spSpec-MJD-PLATE-FIBER.fit``): the primary HDU is a
  4 x Npix image (flux, continuum-subtracted flux, 1-sigma error, mask) with a
  log-linear wavelength solution, lambda_i = 10**(COEFF0 + COEFF1 * i).
* Any 1-D image spectrum with a linear WCS (CRVAL1 / CDELT1 / CRPIX1).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def read_spectrum(filename: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Read a spectrum from a FITS, ASCII or CSV file.

    Parameters
    ----------
    filename : str
        Path to the file. ``.fits``/``.fit`` are read as FITS, ``.csv`` as
        comma-separated text, anything else as whitespace-separated text with
        columns wavelength, flux and (optionally) error.

    Returns
    -------
    wave : ndarray
        Wavelength in Angstrom.
    flux : ndarray
    error : ndarray or None
        1-sigma flux error, if the file has one.
    header : dict
        FITS primary header, or a small dict describing a text file.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".fits", ".fit"):
        return _read_fits(filename)
    if ext == ".csv":
        return _read_ascii(filename, delimiter=",")
    return _read_ascii(filename)


def _read_fits(filename: str):
    from astropy.io import fits

    with fits.open(filename) as hdul:
        header = dict(hdul[0].header)

        for reader in (_read_sdss_dr8plus, _read_sdss_dr7):
            result = reader(hdul, header)
            if result is not None:
                return result

        return _read_generic_fits_image(hdul, header, filename)


def _read_sdss_dr8plus(hdul, header):
    """DR8+ binary table with flux / loglam / ivar columns, else None."""
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

    def _get(name):
        if name in colnames:
            return np.array(data[data.columns.names[colnames.index(name)]], dtype=float)
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
    DR7-and-earlier spSpec image (rows: flux, continuum-subtracted flux,
    noise, mask), else None.
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
    wave = 10 ** (coeff0 + coeff1 * np.arange(npix))

    return wave, flux, error, header


def _read_generic_fits_image(hdul, header, filename):
    """1-D image spectrum with a linear WCS, e.g. non-SDSS FITS files."""
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
    Write a results dictionary to a JSON file. NumPy arrays and scalars are
    converted to plain Python types.
    """

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    with open(filename, "w") as f:
        json.dump(parameters, f, indent=2, default=_default)


def load_results(filename: str) -> Dict[str, Any]:
    """Read back a JSON file written by :func:`save_results`."""
    with open(filename, "r") as f:
        return json.load(f)
