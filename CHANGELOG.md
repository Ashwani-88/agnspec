# Changelog

## 1.0.1

Second release.

Added
* `balmer.BalmerPseudoContinuum`: Kovačević, Popović & Kollatschny (2014) Balmer pseudo-continuum.
* `optical_region.OpticalRegionModel` and `load_park2022_feii`: decomposition of the 4000-5500 A region.
* `plotting.plot_balmer_fit`, `plot_continuum_windows`, `flux_unit_from_header`.
* `utils.get_ebv`: Galactic E(B-V) from the IRSA dust service.
* Example notebook for the Balmer continuum fit.
* LICENSE, CHANGELOG, `.gitignore`.

Fixed
* Optical line wavelengths in `LINE_WAVELENGTHS` were air values but are used with vacuum SDSS spectra; they are now vacuum.
* `preprocessing.to_rest_frame` shifts the wavelength only and no longer rescales flux; `run_pipeline` no longer rescales errors by (1 + z) and calls `deredden_spectrum` correctly.
* `deredden_spectrum` works without the `extinction` package (built-in CCM89) and raises on an unknown law instead of silently using CCM89.
* `HostGalaxy` starts from the non-negative least-squares weights (starting from zero left the fit stuck on the bound).
* `MCMC` and `NestedSampling`: `seed` now makes the run reproducible.
* `virial_blackhole_mass` raises on an unknown calibration instead of silently using the default.
* The Fe II template files are now included when the package is installed with `pip install .`.
* `plot_spectrum` raises on an unknown `stage` instead of silently using raw-spectrum styling.

Removed
* Unused or duplicated code: `fit_powerlaw`, `fit_feii`, `fit_host_galaxy`, `fit_continuum`, `continuum.BalmerContinuum` (superseded by `BalmerPseudoContinuum`), `fit_gaussian`, `fit_multiple_gaussians`, `mcmc_fit`, `trim_spectrum`, `compare_spectra`, `AGNModel.add_continuum_component`, several unused helpers in `utils`, the numpy-only FITS reader, the unfinished `stellar_continuum` module and the data files only it used.
* The unverified `gh05` black-hole-mass calibration.

Planned: example notebooks for MCMC fitting and UV Fe II.

## 0.1.0

First version.
