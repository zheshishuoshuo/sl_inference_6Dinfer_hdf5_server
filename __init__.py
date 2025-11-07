"""sl_inference_restart package.

Minimal package initialisation exposing the core inference utilities.
"""

# from .run_mcmc import run_mcmc
# from .likelihood import log_prior, log_likelihood, log_posterior
# from .utils import selection_function, mag_likelihood

__all__ = [
    "run_mcmc",
    "log_prior",
    "log_likelihood",
    "log_posterior",
    "selection_function",
    "mag_likelihood",
    "plot_chain",
]

__version__ = "0.1"
