"""
agnspec
=======

A component-oriented Python package for fitting AGN (Active Galactic
Nuclei) optical/UV spectra.

The package is organized around *spectral components* rather than
*fitting steps*:

    AGNModel
    ├── Continuum
    │   ├── PowerLaw
    │   ├── HostGalaxy
    │   ├── BalmerContinuum
    │   └── FeII
    │
    ├── EmissionLines
    │   ├── HbetaComplex
    │   ├── HalphaComplex
    │   ├── MgIIComplex
    │   ├── CIVComplex
    │   └── OIIIComplex
    │
    ├── Fitters
    │   ├── LeastSquares
    │   ├── MCMC
    │   └── NestedSampling
    │
    ├── Measurements
    ├── Plotting
    └── Pipeline

Each physical component (PowerLaw, FeII, HbetaComplex, ...) knows how to
generate its own model flux and how many/which free parameters it has.
The `AGNModel` class simply sums the components together, and any of
the `Fitters` can be attached to optimize the combined parameter
vector against data.
"""

from . import io
from . import preprocessing
from . import continuum
from . import emission_lines
from . import fitting
from . import measurements
from . import uncertainties
from . import plotting
from . import utils
from . import pipeline
from . import models

from .models import AGNModel
from .pipeline import run_pipeline

__all__ = [
    "io", "preprocessing", "continuum", "emission_lines", "fitting",
    "measurements", "uncertainties", "plotting", "utils", "pipeline",
    "models", "AGNModel", "run_pipeline",
]

__version__ = "0.1.0"
