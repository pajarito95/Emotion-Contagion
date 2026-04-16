"""
Build the initial state of the emotion contagion simulation.

Current design assumptions:
- All agents, including the leader, are stored in one shared `agents` list.
- The leader is identified by `leader_index`.
- One unified intimacy matrix stores all pairwise ties.
- Agents are labeled with role = "leader" or role = "member".
- For now, regular contagion updates are applied only to members.
- Leader intervention is handled separately in leader_intervention.py.
- RL is not included here (yet).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from agents import make_agents, configure_leader
from network import create_intimacy_matrix
from simulation_state import SimulationState


VALID_STRUCTURES = {"random", "community", "core_periphery"}
VALID_STYLES = {"No_Intervention", "High_Initially_Constrained", "Low_Initially_Constrained",}


def validate_population_size(population_size: int) -> None:
    if not isinstance(population_size, int):
        raise TypeError(
            f"population_size must be an integer. Received instead: "
            f"{type(population_size).__name__}."
        )
    
    if population_size < 2:
        raise ValueError(
            "population_size must be at least 2 so there is one leader and at least one member."
        )


def validate_structure(structure: str) -> None:
    if structure not in VALID_STRUCTURES:
        raise ValueError(
            f"Invalid structure {structure!r}. Choose from: "
            f"{sorted(VALID_STRUCTURES)}."
        )


def validate_style(style: str) -> None:
    if style not in VALID_STYLES:
        raise ValueError(
            f"Invalid leader style {style!r}. Choose from: "
            f"{sorted(VALID_STYLES)}."
        )


def assign_indices_and_roles(agents: list[dict], leader_index: int) -> list[dict]:
    """
    Assign a stable index and role to every agent.

    Parameters
    ----------
    agents: list[dict]
        All agents including the leader
    leader_index: int
        Index of the leader in the list

    Returns
    -------
    list[dict]
        Updated agents list
    """
    for idx, agent in enumerate(agents):
        agent["index"] = idx
        agent["role"] = "leader" if idx == leader_index else "member"
        
    return agents


def select_and_configure_leader(
    rng: np.random.Generator,
    agents: list[dict],
    leader_style: str,
) -> tuple[list[dict], int]:
    """
    Randomly choose one agent to be leader and configure that agent in place. Leader stays inside agents list.

    Parameters
    ----------
    rng: np.random.Generator
        Random number generator
    agents: list[dict]
        All agents.
    leader_style: str
        Leader style

    Returns
    -------
    tuple[list[dict], int]
        Updated agents list and leader index
    """
    leader_index = int(rng.integers(0, len(agents)))
    leader = agents[leader_index]

    # Remove member-specific fields from the leader if present
    for key in ["delta", "expressiveness", "amplification", "bias"]:
        if key in leader:
            del leader[key]

    leader["emotion"] = 1.0
    leader = configure_leader(leader, leader_style)

    agents[leader_index] = leader

    return agents, leader_index


def build_initial_intimacy_matrix(
    rng: np.random.Generator,
    population_size: int,
    structure: str,
    intra_strength: float,
    inter_strength: float,
    community_size: Optional[Sequence[int]] = None,
    min_weight: float = 0.01,
    core_to_core: float = 0.65,
    core_to_periph: float = 0.5,
    periph_to_core: float = 0.2,
    periph_to_periph: float = 0.2,
    core_proportion: float = 0.25,
    leader_index: Optional[int] = None,
    include_leader_ties: bool = True,
    leader_to_member_cap: Optional[float] = None,
    member_to_leader_cap: Optional[float] = None,
    leader_to_member_value: Optional[float] = None,
    member_to_leader_value: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one unified intimacy matrix over all agents. 
    Base matrix generation is delegated to network.py, then optionally overwrites leader-related ties to make leader-member ties distinct from member-member ties.

    Parameters
    ----------
    rng: np.random.Generator
        Random number generator
    population_size: int
        Total number of agents including the leader
    structure: str
        Network structure
    intra_strength, inter_strength: float
        Follower-style baseline bounds for community/random structures
    community_size: sequence[int] or None, optional
        Optional explicit community sizes
    min_weight: float, optional
        Minimum raw tie weight
    core_to_core, core_to_periph, periph_to_core, periph_to_periph: float, optional
        Core-periphery bounds
    core_proportion: float, optional
        Core proportion
    leader_index: int or None, optional
        Index of leader in the full matrix
    include_leader_ties: bool, optional
        Whether leader-member ties should be active at initialization
    leader_to_member_cap: float or None, optional
        Upper bound for leader -> member random ties
    member_to_leader_cap: float or None, optional
        Upper bound for member -> leader random ties
    leader_to_member_value: float or None, optional
        Fixed value for all leader -> member ties. If provided, this overrides leader_to_member_cap
    member_to_leader_value: float or None, optional
        Fixed value for all member -> leader ties. If provided, this overrides member_to_leader_cap

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Complete intimacy matrix and assignments
    """
    intimacy_matrix, assignments = create_intimacy_matrix(
        rng=rng,
        population=population_size,
        structure=structure,
        intra_strength=intra_strength,
        inter_strength=inter_strength,
        core_to_core=core_to_core,
        core_to_periph=core_to_periph,
        periph_to_core=periph_to_core,
        periph_to_periph=periph_to_periph,
        core_proportion=core_proportion,
        size=community_size,
        min_weight=min_weight,
        leader_index=leader_index,
    )

    if leader_index is None:
        return intimacy_matrix, assignments

    if not include_leader_ties:
        intimacy_matrix[leader_index,:] = 0.0
        intimacy_matrix[:, leader_index] = 0.0
        intimacy_matrix[leader_index, leader_index] = 0.0
        row_sums = intimacy_matrix.sum(axis=1, keepdims=True)
        safe_rows = row_sums.squeeze() > 0
        intimacy_matrix[safe_rows] = (intimacy_matrix[safe_rows] / row_sums[safe_rows])
        return intimacy_matrix, assignments

    n = population_size

    for j in range(n):
        if j == leader_index:
            continue

        # leader -> member
        if leader_to_member_value is not None:
            intimacy_matrix[leader_index, j] = leader_to_member_value
        elif leader_to_member_cap is not None:
            if leader_to_member_cap < min_weight:
                raise ValueError("leader_to_member_cap must be at least min_weight.")
            intimacy_matrix[leader_index, j] = rng.uniform(min_weight, leader_to_member_cap)

        # member -> leader
        if member_to_leader_value is not None:
            intimacy_matrix[j, leader_index] = member_to_leader_value
        elif member_to_leader_cap is not None:
            if member_to_leader_cap < min_weight:
                raise ValueError("member_to_leader_cap must be at least min_weight.")
            intimacy_matrix[j, leader_index] = rng.uniform(min_weight, member_to_leader_cap)

    intimacy_matrix[leader_index, leader_index] = 0.0

    row_sums = intimacy_matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("At least one intimacy row has zero sum after applying leader tie rules. Increase allowed tie values or avoid zeroing all ties in a row.")

    intimacy_matrix = intimacy_matrix / row_sums
    return intimacy_matrix, assignments


def initialize_simulation(
    rng: np.random.Generator,
    population_size: int,
    structure: str,
    leader_style: str,
    intra_strength: float = 0.5,
    inter_strength: float = 0.2,
    community_size: Optional[Sequence[int]] = None,
    min_weight: float = 0.01,
    core_to_core: float = 0.65,
    core_to_periph: float = 0.5,
    periph_to_core: float = 0.2,
    periph_to_periph: float = 0.2,
    core_proportion: float = 0.25,
    include_leader_ties: bool = True,
    leader_to_member_cap: Optional[float] = None,
    member_to_leader_cap: Optional[float] = None,
    leader_to_member_value: Optional[float] = None,
    member_to_leader_value: Optional[float] = None,
) -> SimulationState:
    """
    Create the full initial simulation state.
    1. Create all agents
    2. Randomly choose one to become leader
    3. Configure that leader in place
    4. Assign indices and roles
    5. Build one unified intimacy matrix over all agents
    6. Return a SimulationState object

    Returns
    -------
    SimulationState
        Initial state at time 0
    """
    validate_population_size(population_size)
    validate_structure(structure)
    validate_style(leader_style)

    agents = make_agents(rng, population_size)

    if len(agents) != population_size:
        raise ValueError(
            f"make_agents returned {len(agents)} agents, but "
            f"population_size={population_size} was requested."
        )

    agents, leader_index = select_and_configure_leader(rng=rng, agents=agents, leader_style=leader_style)

    agents = assign_indices_and_roles(agents=agents, leader_index=leader_index)

    intimacy_matrix, assignments = build_initial_intimacy_matrix(
        rng=rng,
        population_size=population_size,
        structure=structure,
        intra_strength=intra_strength,
        inter_strength=inter_strength,
        community_size=community_size,
        min_weight=min_weight,
        core_to_core=core_to_core,
        core_to_periph=core_to_periph,
        periph_to_core=periph_to_core,
        periph_to_periph=periph_to_periph,
        core_proportion=core_proportion,
        leader_index=leader_index,
        include_leader_ties=include_leader_ties,
        leader_to_member_cap=leader_to_member_cap,
        member_to_leader_cap=member_to_leader_cap,
        leader_to_member_value=leader_to_member_value,
        member_to_leader_value=member_to_leader_value,
    )

    metadata = {
        "population_size": population_size,
        "structure": structure,
        "leader_style": leader_style,
        "include_leader_ties": include_leader_ties,
        "leader_to_member_cap": leader_to_member_cap,
        "member_to_leader_cap": member_to_leader_cap,
        "leader_to_member_value": leader_to_member_value,
        "member_to_leader_value": member_to_leader_value,
    }

    return SimulationState(
        agents=agents,
        leader_index=leader_index,
        intimacy_matrix=intimacy_matrix,
        assignments=assignments,
        time=0,
        metadata=metadata,
    )