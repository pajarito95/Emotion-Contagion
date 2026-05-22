"""
running package

Convenience imports for simulation execution entry points.
"""

from .run_simulation import SimulationResults, run_simulation
from .run_multiple_simulations import build_conditions_from_grid, run_multiple_simulations

__all__ = [
    "SimulationResults",
    "run_simulation",
    "build_conditions_from_grid",
    "run_multiple_simulations",
]
