"""
Build directed, row-normalized intimacy matrices for the emotion contagion ABM.

Current design assumptions:
- All agents, including the leader, are part of one shared intimacy matrix.
- The leader is identified by `leader_index`.
- Each network structure has its own natural parameterization.
- The matrix is directed/asymmetric and row-normalized.

Structure-specific inputs:
- random:
    - strength
- community:
    - n_communities
    - intra_strength
    - inter_strength
- core_periphery:
    - core_proportion
    - core_to_core
    - core_to_periph
    - periph_to_core
    - periph_to_periph
"""

from __future__ import annotations
from typing import Optional
import numpy as np

VALID_STRUCTURES = {"community", "random", "core_periphery"}

def _validate_common_inputs(population: int, structure: str, leader_index: Optional[int]) -> None:
    if not isinstance(population, int):
        raise TypeError(f"population must be an integer, but received {type(population).__name__}.")

    if population < 2:
        raise ValueError("population must be at least 2 so the simulation has one leader and at least one member.")

    if structure not in VALID_STRUCTURES:
        raise ValueError(f"Invalid structure {structure!r}. Choose from: {sorted(VALID_STRUCTURES)}.")

    if leader_index is None:
        raise ValueError("leader_index must be provided because the leader is part of the shared intimacy matrix.")

    if leader_index != population - 1:
        raise ValueError(f"leader_index must be the last index in the population list (population - 1), but received {leader_index}.")

    if not isinstance(leader_index, int):
        raise TypeError(f"leader_index must be an integer, but received {type(leader_index).__name__}.")

    if not (0 <= leader_index < population):
        raise ValueError(f"leader_index={leader_index} is out of bounds for population={population}.")

# UPDATED
def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = np.abs(matrix).sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("At least one row of the intimacy matrix has zero absolute sum before normalization. Check network generation parameters.")
    return matrix / row_sums

# UPDATED
def _sample_signed_weight(rng: np.random.Generator, lower_bound: float = -1.0, upper_bound: float = 1.0) -> float:
    if not (-1.0 <= lower_bound <= 1.0 and -1.0 <= upper_bound <= 1.0):
        raise ValueError(f"Signed weights must stay within [-1, 1], but received [{lower_bound}, {upper_bound}].")
    
    weight = float(rng.normal(loc=0.5, scale=0.25))  # normal distribution centered at 0.5 with std dev of 0.25, no clipping
    # weight = np.clip(weight, -1.0, 1.0)
    return weight

def _community_assignments(rng: np.random.Generator, population: int, n_communities: int) -> np.ndarray:
    """
    Create assignments for a community structure over the full population.

    Returns an integer array of length population with labels in {0, ..., n_communities - 1}.
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

def core_probability_matrix(
        n_peripheries: int,
        core_to_core: float,
        core_to_periph: float,
        periph_to_core: float,
        periph_to_periph: float,
        directed=True
        ) -> np.ndarray:

    """
    Build probability matrix M for core periphery construction. 
    Off-diagonal values are zero (between periphery probs = 0)
    """
    if directed == False:
        assert core_to_periph == periph_to_core, "Undirected matrix expects symmetric probabilities for core-to-periph and periph-to-core."

    # set size of matrix to be a square of the number of cores and peripheries total
    M = np.zeros((1 + n_peripheries, 1 + n_peripheries), dtype=float)

    # fill core-to-core probability (won't be used for edge-creation if no self-looping if core is of size 1)
    M[0,0] = core_to_core

    # fill core-to-periphery probability
    M[0,1:] = core_to_periph

    # fill periphery-to-core probability
    M[1:,0] = periph_to_core

    # fill periphery-to-periphery probability (the non-core diagonal; within-periphery)
    np.fill_diagonal(M[1:,1:], periph_to_periph)

    return M

def _core_periphery_assignments(
    rng: np.random.Generator, 
    population: int, 
    core_proportion: float, 
    n_peripheries: int,
    leader_index: int, 
    include_leader_ties: bool = False,
    ) -> np.ndarray:
    """
    Create random core-periphery assignments.

    Returns an integer array of length population with labels:
    - 1 = core
    - 0 = periphery
    """
    # validations here...

    assignments = np.zeros(population, dtype=int)  # leader will stay 0, others will get assigned a number
    if include_leader_ties: 
        n_per_pblock = (population - 1) // n_peripheries  # determine how many members per periphery -- WAIT, lower_ceiling needed?      
        remainder = (population - 1) % n_peripheries  # how many remaining members are there to be distributed?
        pblock_sizes = [n_per_pblock + (1 if i < remainder else 0) for i in range(n_peripheries)]  # create list containing the sizes of each periphery where the size of the list is how many peripheries there are

        # create array of member indices
        member_indices = np.delete(np.arange(population), leader_index)

        # shuffle the members' indices for randomization of assignments
        shuffled_indices = rng.permutation(member_indices)

        # assign the members to peripheries
        start = 0  # since 0 is leader
        for periphery_id, size in enumerate(pblock_sizes, start=1):
            end = start + size  # to help in selecting portions of `agents` to use
            assignments[shuffled_indices[start:end]] = periphery_id
            start = end  # move the starting point forward to get next chunk of members
    else: 
        core_size = max(1, round(population*core_proportion))
        core_indices = rng.choice(population, size=core_size, replace=False)
        assignments[core_indices] = 1
        
    return assignments

def _iter_pairs(n: int):
    """
    Iterate over each unordered pair of distince nodes once
    """
    for i in range(n):
        for j in range(i+1,n):
            yield i,j

def no_lonelies(W: np.ndarray, rng: np.random.Generator, directed: bool) -> None:
    """
    Ensure every node has at least one outgoing connection
    """
    n = W.shape[0]
    for i in range(n):
        if np.any(W[i] != 0):
            continue

        candidates = [j for j in range(n) if j != i]
        j = rng.choice(candidates)

        if directed:
            W[i, j] = _sample_signed_weight(rng)
            W[j, i] = _sample_signed_weight(rng)
        else:
            value = _sample_signed_weight(rng)
            W[i, j] = value
            W[j, i] = value

def create_intimacy_matrix(
    rng: np.random.Generator,
    population: int,
    structure: str,
    directed: bool,
    
    leader_index: int,
    include_leader_ties: bool,
    
    strength: float,

    n_communities: int,
    intra_strength: float,
    inter_strength: float,

    n_peripheries: int,
    core_proportion: float,
    core_to_core: float,
    core_to_periph: float,
    periph_to_core: float,
    periph_to_periph: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a row-normalized intimacy matrix over all agents.

    Parameters:
        rng : np.random.Generator
            Random number generator
        population : int
            Total number of agents, including the leader
        structure : str
            One of: "random", "community", "core_periphery"
        leader_index : int
            Index of the leader in the shared population list
        include_leader_ties : bool
            Whether to include the leader in the network assignments and allow ties between the leader and members

    Random-specific parameters:
        strength : float
            Upper bound for all off-diagonal ties

    Community-specific parameters:
        n_communities : int
            Number of communities to generate
        intra_strength : float
            Upper bound for within-community ties
        inter_strength : float
            Upper bound for between-community ties

    Core-periphery-specific parameters:
        core_proportion : float
            Proportion of agents that will be in the core
        core_to_core: float
            Upper bound for core -> core ties
        core_to_periph : float
            Upper bound for leader/core -> periphery ties
        periph_to_core : float
            Upper bound for periphery -> leader/core ties
        periph_to_periph : float
            Upper bound for periphery -> periphery ties

    Returns:
        tuple[np.ndarray, np.ndarray]
            intimacy_matrix : np.ndarray
                Full NxN directed, row-normalized intimacy matrix
            assignments : np.ndarray
                Structure assignments.
                - random/community: integer community labels
                - core_periphery: 1 = core, 0 = periphery
    """
    _validate_common_inputs(population=population, structure=structure, leader_index=leader_index)

    network_population = population if include_leader_ties else population - 1
    if structure == "random":
        assignments = np.zeros(network_population, dtype=int)
    elif structure == "community":
        if n_communities is None:  raise ValueError("For structure='community', you must provide n_communities.")
        assignments = _community_assignments(rng=rng, population=network_population, n_communities=n_communities)
    elif structure == "core_periphery":
        assignments = _core_periphery_assignments(rng, network_population, core_proportion, n_peripheries, leader_index, include_leader_ties)
    else:
        raise ValueError(f"Unhandled structure {structure!r}.")

    W = np.zeros((network_population, network_population), dtype=float)
    M = core_probability_matrix(n_peripheries, core_to_core, core_to_periph, periph_to_core, periph_to_periph, directed)

    for i, j in _iter_pairs(network_population):
        if structure == "random":
            edge_probability = strength
            if rng.random() < edge_probability:
                if directed:
                    W[i,j] = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)
                    W[j,i] = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)
                else:
                    value = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)
                    W[i,j] = value
                    W[j,i] = value
        elif structure == "community":
            same_community = assignments[i] == assignments[j]
            edge_probability = intra_strength if same_community else inter_strength
            if rng.random() < edge_probability:
                if directed:
                    W[i,j] = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)
                    W[j,i] = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)
                else:
                    value = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)
                    W[i,j] = value
                    W[j,i] = value
        elif structure == "core_periphery":
            if directed:
                probability_ij = M[assignments[i], assignments[j]]
                if rng.random() < probability_ij:
                    W[i,j] = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)

                probability_ji = M[assignments[j], assignments[i]]
                if rng.random() < probability_ji:
                    W[j,i] = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)
            else:
                edge_probability = M[assignments[i], assignments[j]]
                if rng.random() < edge_probability:
                    value = _sample_signed_weight(rng=rng, lower_bound=-1.0, upper_bound=1.0)
                    W[i,j] = value
                    W[j,i] = value

    np.fill_diagonal(W, 0.0)
    no_lonelies(W, rng, directed)

    return _normalize_rows(W), assignments