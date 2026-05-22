"""
Build directed, row-normalized intimacy matrices for the emotion contagion ABM.

Current design assumptions:
- All agents, including leader, are part of one shared intimacy matrix.
- Leader is identified by `leader_index`.
- Each network structure has its own natural parameterization.
- Matrix is directed/asymmetric and row-normalized.

Structure-specific inputs:
- random:
    - strength
- community:
    - n_communities
    - intra_strength
    - inter_strength
- core_periphery:
    - leader is the core
    - n_periphery
    - core_to_periph
    - periph_to_core
    - periph_to_periph
"""

from __future__ import annotations

from typing import Optional
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
        raise ValueError("population must be at least 2 so the simulation has one leader and at least one member.")

    if structure not in VALID_STRUCTURES:
        raise ValueError(f"Invalid structure {structure!r}. Choose from: {sorted(VALID_STRUCTURES)}.")

    if not (0 < min_weight < 1):
        raise ValueError(f"min_weight must be between 0 and 1, but received {min_weight}.")

    if leader_index is None:
        raise ValueError("leader_index must be provided because the leader is part of the shared intimacy matrix.")

    if not isinstance(leader_index, int):
        raise TypeError(f"leader_index must be an integer, but received {type(leader_index).__name__}.")

    if not (0 <= leader_index < population):
        raise ValueError(f"leader_index={leader_index} is out of bounds for population={population}.")


def _validate_strength(value: float, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, but received {type(value).__name__}.")
    if not (0 <= value <= 1):
        raise ValueError(f"{name} must be between 0 and 1, but received {value}.")


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("At least one row of the intimacy matrix has zero sum before normalization. Increase tie weights or avoid zeroing an entire row.")
    return matrix / row_sums


def _sample_weight(rng: np.random.Generator, upper_bound: float, min_weight: float) -> float:
    if upper_bound < min_weight:
        raise ValueError(f"Upper bound {upper_bound} is smaller than min_weight {min_weight}. Either increase the upper bound or decrease min_weight.")
    return float(rng.uniform(min_weight, upper_bound))


def _community_assignments(
    rng: np.random.Generator,
    population: int,
    n_communities: int,
) -> np.ndarray:
    """
    Create assignments for a community structure over the full population.

    Returns an integer array of length population with labels in {0, ..., n_communities-1}.
    """
    if not isinstance(n_communities, int):
        raise TypeError(f"n_communities must be an integer, but received {type(n_communities).__name__}.")
    if n_communities < 2:
        raise ValueError(f"n_communities must be at least 2, but received {n_communities}.")
    if n_communities > population:
        raise ValueError(f"n_communities={n_communities} cannot exceed population={population}.")

    base = population // n_communities
    remainder = population % n_communities
    sizes = [base + (1 if i < remainder else 0) for i in range(n_communities)]

    shuffled_indices = rng.permutation(population)
    assignments = np.empty(population, dtype=int)

    start = 0
    for community_id, size in enumerate(sizes):
        end = start + size
        assignments[shuffled_indices[start:end]] = community_id
        start = end

    return assignments


def _core_periphery_assignments(
    population: int,
    leader_index: int,
    n_periphery: int,
) -> np.ndarray:
    """
    Create core-periphery assignments with the leader fixed as the only core.

    Returns an integer array of length population with labels:
    - 1 = core
    - 0 = periphery
    """
    if not isinstance(n_periphery, int):
        raise TypeError(f"n_periphery must be an integer, but received {type(n_periphery).__name__}.")

    n_non_leader = population - 1

    if n_periphery < 1:
        raise ValueError(f"n_periphery must be at least 1, but received {n_periphery}.")

    if n_periphery > n_non_leader:
        raise ValueError(f"Requested n_periphery={n_periphery}, but only {n_non_leader} non-leader agents are available. Reduce n_periphery or increase population.")

    if n_periphery != n_non_leader:
        raise ValueError(f"With the current design, the leader is the single core and all non-leader agents are peripheral, so n_periphery must equal {n_non_leader}. Received {n_periphery}.")

    assignments = np.zeros(population, dtype=int)
    assignments[leader_index] = 1

    return assignments


def create_intimacy_matrix(
    rng: np.random.Generator,
    population: int,
    structure: str,
    min_weight: float = 0.01,
    leader_index: Optional[int] = None,
    strength: Optional[float] = None,
    n_communities: Optional[int] = None,
    intra_strength: Optional[float] = None,
    inter_strength: Optional[float] = None,
    n_periphery: Optional[int] = None,
    core_to_periph: Optional[float] = None,
    periph_to_core: Optional[float] = None,
    periph_to_periph: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an asymmetric, row-normalized intimacy matrix over all agents.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    population : int
        Total number of agents, including the leader.
    structure : str
        One of: "random", "community", "core_periphery".
    min_weight : float, optional
        Minimum raw tie weight before normalization.
    leader_index : int
        Index of the leader in the shared population list.

    Random-specific parameters
    --------------------------
    strength : float
        Single upper bound used for all off-diagonal ties.

    Community-specific parameters
    -----------------------------
    n_communities : int
        Number of communities to generate.
    intra_strength : float
        Upper bound for within-community ties.
    inter_strength : float
        Upper bound for between-community ties.

    Core-periphery-specific parameters
    ----------------------------------
    n_periphery : int
        Number of peripheral agents. Under the current design this must equal the number of non-leader agents.
    core_to_periph : float
        Upper bound for leader/core -> periphery ties.
    periph_to_core : float
        Upper bound for periphery -> leader/core ties.
    periph_to_periph : float
        Upper bound for periphery -> periphery ties.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        intimacy_matrix : np.ndarray
            Full NxN directed, row-normalized intimacy matrix.
        assignments : np.ndarray
            Structure assignments.
            - random/community: integer community labels
            - core_periphery: 1 = core, 0 = periphery
    """
    _validate_common_inputs(
        population=population,
        structure=structure,
        min_weight=min_weight,
        leader_index=leader_index,
    )

    W = np.zeros((population, population), dtype=float)

    if structure == "random":
        if strength is None:
            raise ValueError("For structure='random', you must provide strength.")
        _validate_strength(strength, "strength")

        assignments = np.zeros(population, dtype=int)

        for i in range(population):
            for j in range(population):
                if i == j:
                    continue
                W[i, j] = _sample_weight(rng=rng, upper_bound=strength, min_weight=min_weight)

        np.fill_diagonal(W, 0.0)
        return _normalize_rows(W), assignments

    if structure == "community":
        if n_communities is None:
            raise ValueError("For structure='community', you must provide n_communities.")
        if intra_strength is None:
            raise ValueError("For structure='community', you must provide intra_strength.")
        if inter_strength is None:
            raise ValueError("For structure='community', you must provide inter_strength.")

        _validate_strength(intra_strength, "intra_strength")
        _validate_strength(inter_strength, "inter_strength")

        assignments = _community_assignments(
            rng=rng,
            population=population,
            n_communities=n_communities,
        )

        for i in range(population):
            for j in range(population):
                if i == j:
                    continue

                upper = intra_strength if assignments[i] == assignments[j] else inter_strength
                W[i, j] = _sample_weight(rng=rng, upper_bound=upper, min_weight=min_weight)

        np.fill_diagonal(W, 0.0)
        return _normalize_rows(W), assignments

    if structure == "core_periphery":
        if n_periphery is None:
            raise ValueError("For structure='core_periphery', you must provide n_periphery.")
        if core_to_periph is None:
            raise ValueError("For structure='core_periphery', you must provide core_to_periph.")
        if periph_to_core is None:
            raise ValueError("For structure='core_periphery', you must provide periph_to_core.")
        if periph_to_periph is None:
            raise ValueError("For structure='core_periphery', you must provide periph_to_periph.")

        _validate_strength(core_to_periph, "core_to_periph")
        _validate_strength(periph_to_core, "periph_to_core")
        _validate_strength(periph_to_periph, "periph_to_periph")

        assignments = _core_periphery_assignments(
            population=population,
            leader_index=leader_index,
            n_periphery=n_periphery,
        )

        for i in range(population):
            for j in range(population):
                if i == j:
                    continue

                from_is_core = assignments[i] == 1
                to_is_core = assignments[j] == 1

                if from_is_core and not to_is_core:
                    upper = core_to_periph
                elif not from_is_core and to_is_core:
                    upper = periph_to_core
                elif not from_is_core and not to_is_core:
                    upper = periph_to_periph
                else:
                    raise ValueError("core_to_core ties are not used in the current core_periphery design because the leader is the single core.")

                W[i, j] = _sample_weight(rng=rng, upper_bound=upper, min_weight=min_weight)

        np.fill_diagonal(W, 0.0)
        return _normalize_rows(W), assignments

    raise ValueError(f"Unhandled structure {structure!r}. This should not happen after validation.")