"""
Build the initial state of the emotion contagion simulation.

Current design assumptions:
- All agents, including the leader, are stored in one shared `agents` list.
- The leader is identified by `leader_index`.
- One unified intimacy matrix stores all pairwise ties.
- Agents are labeled with role = "leader" or role = "member".
- For now, regular contagion updates are applied only to members.
- Leader intervention is handled separately in leader_intervention.py.
- RL is not included here yet.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agents import make_agents, configure_leader
from network import create_intimacy_matrix
from simulation_state import SimulationState


VALID_STRUCTURES = {"random", "community", "core_periphery"}
VALID_STYLES = {"No_Intervention", "High_Initially_Constrained", "Low_Initially_Constrained", "High_Fully_Constrained", "Low_Fully_Constrained", "Free"}


def validate_population_size(population_size: int) -> None:
    if not isinstance(population_size, int):
        raise TypeError(f"population_size must be an integer. Received instead: {type(population_size).__name__}.")
    if population_size < 2:
        raise ValueError("population_size must be at least 2 so there is one leader and at least one member.")


def validate_structure(structure: str) -> None:
    if structure not in VALID_STRUCTURES:
        raise ValueError(f"Invalid structure {structure!r}. Choose from: {sorted(VALID_STRUCTURES)}.")


def validate_style(style: str) -> None:
    if style not in VALID_STYLES:
        raise ValueError(f"Invalid leader style {style!r}. Choose from: {sorted(VALID_STYLES)}.")


def assign_indices_and_roles(agents: list[dict], leader_index: int) -> list[dict]:
    for idx, agent in enumerate(agents):
        agent["index"] = idx
        agent["role"] = "leader" if idx == leader_index else "member"
    return agents


def select_and_configure_leader(
    rng: np.random.Generator,
    agents: list[dict],
    leader_style: str,
) -> tuple[list[dict], int]:
    leader_index = int(rng.integers(0, len(agents)))
    leader = agents[leader_index]

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
    leader_index: int,
    min_weight: float = 0.01,
    include_leader_ties: bool = True,
    leader_to_member_cap: Optional[float] = None,
    member_to_leader_cap: Optional[float] = None,
    leader_to_member_value: Optional[float] = None,
    member_to_leader_value: Optional[float] = None,
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
    Build one unified intimacy matrix over all agents.

    Base matrix generation is delegated to network.py, then optionally leader-related ties are overwritten to make leader-member ties distinct from member-member ties.
    """
    intimacy_matrix, assignments = create_intimacy_matrix(
        rng=rng,
        population=population_size,
        structure=structure,
        min_weight=min_weight,
        leader_index=leader_index,
        strength=strength,
        n_communities=n_communities,
        intra_strength=intra_strength,
        inter_strength=inter_strength,
        n_periphery=n_periphery,
        core_to_periph=core_to_periph,
        periph_to_core=periph_to_core,
        periph_to_periph=periph_to_periph,
    )

    if not include_leader_ties:
        intimacy_matrix[leader_index, :] = 0.0
        intimacy_matrix[:, leader_index] = 0.0
        intimacy_matrix[leader_index, leader_index] = 0.0

        row_sums = intimacy_matrix.sum(axis=1, keepdims=True)
        safe_rows = row_sums.squeeze() > 0
        intimacy_matrix[safe_rows] = intimacy_matrix[safe_rows] / row_sums[safe_rows]
        return intimacy_matrix, assignments

    n = population_size

    for j in range(n):
        if j == leader_index:
            continue

        if leader_to_member_value is not None:
            intimacy_matrix[leader_index, j] = leader_to_member_value
        elif leader_to_member_cap is not None:
            if leader_to_member_cap < min_weight:
                raise ValueError("leader_to_member_cap must be at least min_weight.")
            intimacy_matrix[leader_index, j] = rng.uniform(min_weight, leader_to_member_cap)

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
    min_weight: float = 0.01,
    include_leader_ties: bool = True,
    leader_to_member_cap: Optional[float] = None,
    member_to_leader_cap: Optional[float] = None,
    leader_to_member_value: Optional[float] = None,
    member_to_leader_value: Optional[float] = None,
    strength: Optional[float] = None,
    n_communities: Optional[int] = None,
    intra_strength: Optional[float] = None,
    inter_strength: Optional[float] = None,
    n_periphery: Optional[int] = None,
    core_to_periph: Optional[float] = None,
    periph_to_core: Optional[float] = None,
    periph_to_periph: Optional[float] = None,
) -> SimulationState:
    """
    Create the full initial simulation state.

    Structure-specific parameters:
    - random: strength
    - community: n_communities, intra_strength, inter_strength
    - core_periphery: n_periphery, core_to_periph, periph_to_core, periph_to_periph
    """
    validate_population_size(population_size)
    validate_structure(structure)
    validate_style(leader_style)

    agents = make_agents(rng, population_size)

    if len(agents) != population_size:
        raise ValueError(f"make_agents returned {len(agents)} agents, but population_size={population_size} was requested.")

    agents, leader_index = select_and_configure_leader(
        rng=rng,
        agents=agents,
        leader_style=leader_style,
    )

    agents = assign_indices_and_roles(
        agents=agents,
        leader_index=leader_index,
    )

    intimacy_matrix, assignments = build_initial_intimacy_matrix(
        rng=rng,
        population_size=population_size,
        structure=structure,
        leader_index=leader_index,
        min_weight=min_weight,
        include_leader_ties=include_leader_ties,
        leader_to_member_cap=leader_to_member_cap,
        member_to_leader_cap=member_to_leader_cap,
        leader_to_member_value=leader_to_member_value,
        member_to_leader_value=member_to_leader_value,
        strength=strength,
        n_communities=n_communities,
        intra_strength=intra_strength,
        inter_strength=inter_strength,
        n_periphery=n_periphery,
        core_to_periph=core_to_periph,
        periph_to_core=periph_to_core,
        periph_to_periph=periph_to_periph,
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
        "strength": strength,
        "n_communities": n_communities,
        "intra_strength": intra_strength,
        "inter_strength": inter_strength,
        "n_periphery": n_periphery,
        "core_to_periph": core_to_periph,
        "periph_to_core": periph_to_core,
        "periph_to_periph": periph_to_periph,
    }

    return SimulationState(
        agents=agents,
        leader_index=leader_index,
        intimacy_matrix=intimacy_matrix,
        assignments=assignments,
        time=0,
        metadata=metadata,
    )