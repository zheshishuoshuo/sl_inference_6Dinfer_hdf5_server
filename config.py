"""Global configuration for simulation and inference parameters.

This module also defines project-wide constants for parameters that are treated
as known. In particular, ``sigma_DM`` is now considered known (measurable)
and fixed at a constant value for all inference.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScatterConfig:
    """Measurement scatter settings for observables."""

    star: float = 0.1  # Scatter on log stellar mass [dex]
    mag: float = 0.1    # Scatter on observed magnitudes [mag]


# Global scatter configuration used throughout the package
SCATTER = ScatterConfig()

# Fixed intrinsic scatter in the halo–mass relation, treated as known.
# This replaces any previous inference over ``sigma_DM``.
SIGMA_DM: float = 0.37
