"""
Build initial state of the emotion contagion simulation.

Current design assumptions:
- All agents, including the leader, are stored in one shared `agents` list.
- The leader is identified by the last element in the `agents` list.
- One unified intimacy matrix stores all pairwise ties.
- Agents are labeled with role = "leader" or role = "member".
- For now, regular contagion updates are applied only to members.
- Leader intervention is handled separately in leader_intervention.py.
- RL is included.
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

def select_and_configure_leader(rng: np.random.Generator, agents: list[dict], leader_style: str) -> tuple[list[dict], int]:
    leader_og_index = int(rng.integers(0, len(agents)))  # use this to randomly select a leader from agents list
    leader = agents.pop(leader_og_index)
    leader = configure_leader(leader, leader_style)
    agents.append(leader)  # add leader back to the end of the agents list so it stays within, and agent-matrix indexing still aligns exactly and is easy to iterate over and the last index can easily just be excluded
    leader_index = len(agents) - 1
    return agents, leader_index

def assign_indices_and_roles(agents: list[dict], leader_index: int) -> list[dict]:
    for idx, agent in enumerate(agents):
        agent["index"] = idx
        agent["role"] = "leader" if idx == leader_index else "member"
    return agents

def build_initial_intimacy_matrix(
    rng: np.random.Generator,
    population_size: int,
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
    periph_to_periph: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one unified intimacy matrix over all member agents. Base matrix generation is delegated to network.py.
    Leader is excluded from network generation and should not appear in any intimacy matrix.
    """
    intimacy_matrix, assignments = create_intimacy_matrix(
        rng=rng,
        population=population_size,
        structure=structure,
        directed=directed,

        leader_index=leader_index,
        include_leader_ties=include_leader_ties,

        strength=strength,

        n_communities=n_communities,
        intra_strength=intra_strength,
        inter_strength=inter_strength,

        n_peripheries=n_peripheries,
        core_proportion=core_proportion,
        core_to_core=core_to_core,
        core_to_periph=core_to_periph,
        periph_to_core=periph_to_core,
        periph_to_periph=periph_to_periph
    )

    return intimacy_matrix, assignments

def initialize_simulation(
    rng: np.random.Generator,
    population_size: int,

    susceptibility: float,
    expressiveness: float,
    amplification: float,
    bias: float,

    structure: str,
    directed: bool,

    leader_style: str,
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
    periph_to_periph: float,
) -> SimulationState:
    """
    Create full initial simulation state.

    Structure-specific parameters:
    - random: strength
    - community: n_communities, intra_strength, inter_strength
    - core_periphery: core_proportion, core_to_core, core_to_periph, periph_to_core, periph_to_periph
    """
    validate_population_size(population_size)
    validate_structure(structure)
    validate_style(leader_style)

    agents = make_agents(rng, population_size, susceptibility, expressiveness, amplification, bias)

    if len(agents) != population_size:
        raise ValueError(f"make_agents returned {len(agents)} agents, but population_size={population_size} was requested.")

    agents, leader_index = select_and_configure_leader(rng=rng, agents=agents, leader_style=leader_style)
    agents = assign_indices_and_roles(agents=agents, leader_index=leader_index)

    intimacy_matrix, assignments = build_initial_intimacy_matrix(
        rng=rng,
        population_size=population_size,
        structure=structure,
        directed=directed,

        leader_index=leader_index,
        include_leader_ties=include_leader_ties,

        strength=strength,

        n_communities=n_communities,
        intra_strength=intra_strength,
        inter_strength=inter_strength,

        n_peripheries=n_peripheries,
        core_proportion=core_proportion,
        core_to_core=core_to_core,
        core_to_periph=core_to_periph,
        periph_to_core=periph_to_core,
        periph_to_periph=periph_to_periph,
    )

    # NOTE: maybe don't track all this?
    metadata = {
        "population_size": population_size,
        "structure": structure,
        "directed": directed,
        "leader_style": leader_style,
        "leader_index": leader_index,
        "include_leader_ties": include_leader_ties,

        "strength": strength,
        "n_communities": n_communities,
        "intra_strength": intra_strength,
        "inter_strength": inter_strength,

        "n_peripheries": n_peripheries,
        "core_proportion": core_proportion,
        "core_to_core": core_to_core,
        "core_to_periph": core_to_periph,
        "periph_to_core": periph_to_core,
        "periph_to_periph": periph_to_periph,
    }

    return SimulationState(agents=agents, leader_index=leader_index, intimacy_matrix=intimacy_matrix, assignments=assignments, time=0, metadata=metadata)