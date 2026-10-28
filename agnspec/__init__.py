"""
agnspec
=======

Fitting tools for AGN optical and UV spectra, built around spectral
components: a continuum made of a power law, Fe II, a Balmer pseudo-continuum
and host-galaxy templates, plus emission-line complexes (H-beta, H-alpha,
Mg II, C IV, [OIII]). Components can be fitted with least squares, MCMC
(emcee) or nested sampling (dynesty).

Author: Ashwani Pandey
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
from . import balmer
from . import optical_region

from .models import AGNModel
from .pipeline import run_pipeline

__all__ = [
    "io", "preprocessing", "continuum", "emission_lines", "fitting",
    "measurements", "uncertainties", "plotting", "utils", "pipeline",
    "models", "balmer", "optical_region", "AGNModel", "run_pipeline",
]

__author__ = "Ashwani Pandey"
__version__ = "1.0.1"
