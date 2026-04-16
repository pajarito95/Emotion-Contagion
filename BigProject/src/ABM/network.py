"""
Build directed, row-normalized intimacy matrices for the emotion contagion ABM.

Current design assumptions:
- All agents, including leader, are part of one shared intimacy matrix.
- Leader is identified by `leader_index`.
- Follower-follower ties are generated according to the chosen network structure.
- Leader-related ties are allowed in the matrix, but can be modified later by build_simulation.py if leader-member ties should follow different rules.
- Matrix is directed/asymmetric and row-normalized.

TODO: Now that the leader is kept in the intimacy matrix, we need to address how to handle its inclusion in the community and core-periphery structures. 
      Currently it's just assigned to a group like any other agent.
"""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np


VALID_STRUCTURES = {"community", "random", "core_periphery"}

def _validate_common_inputs(
    population: int,
    structure: str,
    min_weight: float,
    leader_index: Optional[int],
) -> None:
    if not isinstance(population, int):
        raise TypeError(f"population must be an integer, but received {type(population).__name__}.")

    if population < 2:
        raise ValueError(
            "population must be at least 2 so the simulation has one leader and at least one member."
        )

    if structure not in VALID_STRUCTURES:
        raise ValueError(f"Invalid structure {structure!r}. Choose from: {sorted(VALID_STRUCTURES)}.")

    if not (0 < min_weight < 1):
        raise ValueError(f"min_weight must be between 0 and 1, but received {min_weight}.")

    if leader_index is not None and not (0 <= leader_index < population):
        raise ValueError(f"leader_index={leader_index} is out of bounds for population={population}.")


def _validate_strength(value: float, name: str) -> None:
    if not (0 <= value <= 1):
        raise ValueError(f"{name} must be between 0 and 1, but received {value}.")


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(
            "At least one row of the intimacy matrix has zero sum before normalization. Increase tie weights or avoid zeroing an entire row.")
    return matrix / row_sums


def _sample_weight(rng: np.random.Generator, upper_bound: float, min_weight: float) -> float:
    if upper_bound < min_weight:
        raise ValueError(f"Upper bound {upper_bound} is smaller than min_weight {min_weight}. Either increase the upper bound or decrease min_weight.")
    return float(rng.uniform(min_weight, upper_bound))


def _community_assignments(
    rng: np.random.Generator,
    population: int,
    size: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Create two-community assignments over the full population.

    Returns
    -------
    np.ndarray
        Array of length population with labels {0, 1}
    """
    if size is None:
        n0 = population // 2
        n1 = population - n0
    else:
        if len(size) != 2:
            raise ValueError(f"For community structure, size must have length 2, but received {size}.")
        n0, n1 = size
        if n0 + n1 != population:
            raise ValueError(f"Community sizes must sum to population={population}, but received {size}.")

    assignments = np.ones(population, dtype=int)
    group0_indices = rng.choice(population, size=n0, replace=False)
    assignments[group0_indices] = 0

    return assignments


def _core_periphery_assignments(
    rng: np.random.Generator,
    population: int,
    core_proportion: float,
) -> np.ndarray:
    """
    Create core-periphery assignments over the full population.

    Returns
    -------
    np.ndarray
        Array of length population with labels {0, 1}, where 1 = core and 0 = periphery.
    """
    _validate_strength(core_proportion, "core_proportion")

    core_size = max(1, round(population * core_proportion))
    core_indices = rng.choice(population, size=core_size, replace=False)

    assignments = np.zeros(population, dtype=int)
    assignments[core_indices] = 1

    return assignments


def create_intimacy_matrix(
    rng: np.random.Generator,
    population: int,
    structure: str,
    intra_strength: float,
    inter_strength: float,
    core_to_core: float = 0.65,
    core_to_periph: float = 0.5,
    periph_to_core: float = 0.2,
    periph_to_periph: float = 0.2,
    core_proportion: float = 0.25,
    size: Optional[Sequence[int]] = None,
    min_weight: float = 0.01,
    leader_index: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an asymmetric, row-normalized intimacy matrix over all agents.

    Parameters
    ----------
    rng: np.random.Generator
        Random number generator
    population: int
        Total number of agents, including the leader
    structure: str
        One of: "community", "random", "core_periphery"
    intra_strength: float
        Within-group upper bound for community/random structures
    inter_strength: float
        Between-group upper bound for community/random structures
    core_to_core: float, optional
        Upper bound for core -> core ties in core-periphery structure
    core_to_periph: float, optional
        Upper bound for core -> periphery ties
    periph_to_core: float, optional
        Upper bound for periphery -> core ties
    periph_to_periph: float, optional
        Upper bound for periphery -> periphery ties
    core_proportion: float, optional
        Fraction of agents assigned to the core
    size: sequence[int] or None, optional
        Community sizes for two-community structure
    min_weight: float, optional
        Minimum raw tie weight before normalization
    leader_index: int or None, optional
        Index of the leader. Currently used for validation/consistency only
        Leader-related tie customization is handled in build_simulation.py

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        intimacy_matrix: np.ndarray
            Full NxN directed, row-normalized intimacy matrix
        assignments: np.ndarray
            Structure assignments:
            - community/random: {0, 1}
            - core_periphery: {0, 1} with 1 = core, 0 = periphery
    """
    _validate_common_inputs(
        population=population,
        structure=structure,
        min_weight=min_weight,
        leader_index=leader_index,
    )

    _validate_strength(intra_strength, "intra_strength")
    _validate_strength(inter_strength, "inter_strength")
    _validate_strength(core_to_core, "core_to_core")
    _validate_strength(core_to_periph, "core_to_periph")
    _validate_strength(periph_to_core, "periph_to_core")
    _validate_strength(periph_to_periph, "periph_to_periph")

    if structure == "random" and intra_strength != inter_strength:
        raise ValueError("For structure='random', intra_strength and inter_strength must match.")

    if structure in {"community", "random"}:
        assignments = _community_assignments(rng=rng, population=population, size=size)

        block_bounds = np.array([[intra_strength, inter_strength], [inter_strength, intra_strength]], dtype=float)

        W = np.zeros((population, population), dtype=float)

        for i in range(population):
            for j in range(population):
                if i == j:
                    continue

                upper = block_bounds[assignments[i], assignments[j]]
                W[i, j] = _sample_weight(rng=rng, upper_bound=upper, min_weight=min_weight)

        np.fill_diagonal(W, 0.0)
        intimacy_matrix = _normalize_rows(W)

        return intimacy_matrix, assignments

    if structure == "core_periphery":
        assignments = _core_periphery_assignments(rng=rng, population=population, core_proportion=core_proportion)

        # 1 = core, 0 = periphery
        block_bounds = np.array([[periph_to_periph, periph_to_core], [core_to_periph, core_to_core]], dtype=float)

        W = np.zeros((population, population), dtype=float)

        for i in range(population):
            for j in range(population):
                if i == j:
                    continue

                upper = block_bounds[assignments[i], assignments[j]]
                W[i, j] = _sample_weight(rng=rng, upper_bound=upper, min_weight=min_weight)

        np.fill_diagonal(W, 0.0)
        intimacy_matrix = _normalize_rows(W)

        return intimacy_matrix, assignments

    raise ValueError(f"Unhandled structure {structure!r}. This should not happen after validation.")