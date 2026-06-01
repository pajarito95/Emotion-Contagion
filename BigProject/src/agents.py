"""
Agent creation and leader configuration for the emotion contagion ABM.

Current design assumptions:
- All agents stored as dictionaries.
- One agent is later designated as the leader, but remains in the shared agents list.
- Leader is configured in place by changing/removing member-specific parameters and adding leader-specific parameters.
- Network generation, simulation dynamics, and intervention logic are handled in other modules.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

VALID_STYLES = {"No_Intervention", "High_Initially_Constrained", "Low_Initially_Constrained", "High_Fully_Constrained", "Low_Fully_Constrained", "Free"}

def validate_population_size(population_size: int) -> None:
    """
    Validate requested population size.

    Parameters:
        population_size : int
            Total number of agents including the future leader

    Raises:
        `TypeError` if population_size is not an integer
        `ValueError` if population_size is too small
    """
    if not isinstance(population_size, int):
        raise TypeError(f"population_size must be an integer, but received {type(population_size).__name__}.")

    if population_size < 2:
        raise ValueError("population_size must be at least 2 so the simulation has one leader and at least one member.")

def validate_style(style: str) -> None:
    """
    Validate leader style.

    Parameters:
        style : str
            Leader style name.

    Raises:
        `ValueError` if style is unsupported.
    """
    if style not in VALID_STYLES:
        raise ValueError(f"Invalid leader style {style!r}. Choose from: {sorted(VALID_STYLES)}.")

def make_agents(rng: np.random.Generator, population_size: int) -> List[Dict[str, Any]]:
    """
    Create the initial population of agents. Each agent is initialized with member-style parameters. One of these agents may later be reconfigured as the leader.

    Member parameters:
    emotion : float
        Initial emotional valence in approximately [-0.5, 0.5], sampled using a shifted beta distribution to skew more negative
    delta : float
        Susceptibility to influence
    expressiveness : float
        Degree to which an agent outwardly expresses emotion
    amplification : float
        Amplification-related parameter used in the Bosse-style update
    bias : float
        Bias-related parameter used in the Bosse-style update

    NOTE: May want to reconsider the beta distribution

    Parameters:
        rng : np.random.Generator
            Random number generator
        population_size : int
            Total number of agents to create

    Returns:
        list[dict]
            List of agent dictionaries
    """
    validate_population_size(population_size)
    agents = []
    for _ in range(population_size):
        agent = {
            "emotion": -0.5 + rng.beta(2, 5),
            "delta": rng.uniform(0.0, 1.0),
            "expressiveness": rng.uniform(0.0, 1.0),
            "amplification": rng.uniform(0.0, 1.0),
            "bias": rng.uniform(0.0, 1.0),
        }
        agents.append(agent)
    return agents

def configure_leader(leader: Dict[str, Any], style: str) -> Dict[str, Any]:
    """
    Configure one existing agent dictionary as the leader.

    Updates chosen agent in place by:
    - removing member-only fields that leader does not use
    - fixing leader emotion to 1.0
    - adding leader-specific parameters

    Parameters:
        leader : dict
            Agent dictionary chosen to become the leader
        style : str
            Leader style

    Returns:
        dict
            Updated leader dictionary
    """
    validate_style(style)

    if not isinstance(leader, dict):
        raise TypeError(f"leader must be a dictionary, but received {type(leader).__name__}.")

    # Remove member-specific parameters if present
    for key in ["delta", "expressiveness", "amplification", "bias"]:
        if key in leader:
            del leader[key]

    # Base leader settings
    leader["emotion"] = 1.0

    if style == "High_Initially_Constrained" or style == "High_Fully_Constrained":
        leader["emotionManagementAbility"] = "High"
        leader["interventionThreshold"] = -0.5

    elif style == "Low_Initially_Constrained" or style == "Low_Fully_Constrained":
        leader["emotionManagementAbility"] = "Low"
        leader["interventionThreshold"] = -0.7

    elif style == "No_Intervention":
        leader["emotionManagementAbility"] = "None"
        leader["interventionThreshold"] = None

    return leader

def validate_agents(agents: List[Dict[str, Any]]) -> None:
    """
    Validate that the agent list has the expected basic structure.

    Parameters:
        agents : list[dict]
            Agent list

    Raises:
        `TypeError` if agents is not a list of dictionaries.
        `ValueError` if agents is empty.
    """
    if not isinstance(agents, list):
        raise TypeError(f"agents must be a list, but received {type(agents).__name__}.")

    if len(agents) == 0:
        raise ValueError("agents cannot be empty.")

    for i, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise TypeError(f"agents[{i}] must be a dictionary, but received {type(agent).__name__}.")

def get_leader(agents: List[Dict[str, Any]], leader_index: int) -> Dict[str, Any]:
    """
    Return the leader dictionary from the shared agents list.

    Parameters:
        agents : list[dict]
            Full list of agents
        leader_index : int
            Index of the leader

    Returns:
        dict
            The leader dictionary
    """
    validate_agents(agents)

    if not isinstance(leader_index, int):
        raise TypeError(f"leader_index must be an integer, but received {type(leader_index).__name__}.")

    if not (0 <= leader_index < len(agents)):
        raise ValueError(f"leader_index={leader_index} is out of bounds for {len(agents)} agents.")

    return agents[leader_index]

def get_members(agents: List[Dict[str, Any]], leader_index: int) -> List[Dict[str, Any]]:
    """
    Return the non-leader agents from the shared agents list.

    Parameters:
        agents : list[dict]
            Full list of agents
        leader_index : int
            Index of the leader

    Returns:
        list[dict]
            Member agents only
    """
    validate_agents(agents)

    if not isinstance(leader_index, int):
        raise TypeError(f"leader_index must be an integer, but received {type(leader_index).__name__}.")

    if not (0 <= leader_index < len(agents)):
        raise ValueError(f"leader_index={leader_index} is out of bounds for {len(agents)} agents.")

    return [agent for i, agent in enumerate(agents) if i != leader_index]