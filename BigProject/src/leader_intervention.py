"""
Leader-specific intervention logic for the emotion contagion ABM.

Current design assumptions:
- All agents, including the leader, are stored in one shared `agents` list.
- The leader is identified by `leader_index`.
- One unified intimacy matrix stores all pairwise ties.
- The leader remains in the shared matrix but is treated differently from members.
- Leader intervention affects only members.
- RL is not included here yet.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


VALID_STYLES = {"No_Intervention", "High_Initially_Constrained", "Low_Initially_Constrained"}


def validate_leader_style(style: str) -> None:
    """
    Validate that the provided leader style is currently supported.
    """
    if style not in VALID_STYLES:
        raise ValueError(f"Unsupported leader style: {style!r}. Supported styles are: {sorted(VALID_STYLES)}.")


def validate_agents(agents: List[dict]) -> None:
    """
    Validate that agents is a non-empty list of dictionaries.
    """
    if not isinstance(agents, list):
        raise TypeError(f"agents must be a list, but received {type(agents).__name__}.")
    
    if len(agents) == 0:
        raise ValueError("agents cannot be empty.")
    
    for i, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise TypeError(f"agents[{i}] must be a dictionary, but received {type(agent).__name__}.")


def validate_leader_index(agents: List[dict], leader_index: int) -> None:
    """
    Validate that leader_index is a valid index and points to the leader.
    """
    validate_agents(agents)

    if not isinstance(leader_index, int):
        raise TypeError(f"leader_index must be an integer, but received {type(leader_index).__name__}.")

    if not (0 <= leader_index < len(agents)):
        raise ValueError(f"leader_index={leader_index} is out of bounds for {len(agents)} agents.")

    if agents[leader_index].get("role") != "leader":
        raise ValueError(f"Agent at leader_index={leader_index} must have role='leader', but found role={agents[leader_index].get('role')!r}.")


def validate_intimacy_matrix(intimacy_matrix: np.ndarray, agents: List[dict]) -> None:
    """
    Validate that the intimacy matrix matches the full agent population.
    """
    if not isinstance(intimacy_matrix, np.ndarray):
        raise TypeError("intimacy_matrix must be a NumPy array.")
    if intimacy_matrix.ndim != 2:
        raise ValueError(f"intimacy_matrix must be 2D, but received shape {intimacy_matrix.shape}.")
    n_agents = len(agents)
    if intimacy_matrix.shape != (n_agents, n_agents):
        raise ValueError(f"intimacy_matrix shape {intimacy_matrix.shape} does not match the required shape ({n_agents}, {n_agents}).")


def get_member_indices(agents: List[dict], leader_index: int) -> List[int]:
    """
    Return the indices of all non-leader member agents.
    """
    validate_leader_index(agents, leader_index)
    return [i for i, agent in enumerate(agents) if i != leader_index and agent.get("role") == "member"]


def should_leader_intervene(style: str, avg_emotional_valence: float, leader: Dict) -> bool:
    """
    Determine whether the leader should intervene at the current timestep.
    """
    validate_leader_style(style)

    if not isinstance(leader, dict):
        raise TypeError(f"leader must be a dictionary, but received {type(leader).__name__}.")

    if style == "No_Intervention":
        return False

    threshold = leader.get("interventionThreshold", None)

    if threshold is None:
        return False

    return avg_emotional_valence <= threshold


def apply_leader_intervention(
    agents: List[dict],
    leader_index: int,
    intimacy_matrix: np.ndarray,
    dampening: float = 0.08,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> List[dict]:
    """
    Apply the leader's intervention to member agents only. Each member is pulled toward the leader's emotion according to:
        dampening * (leader_emotion - member_emotion) * member_delta * intimacy_matrix[leader_index, member_index]
    """
    validate_leader_index(agents, leader_index)
    validate_intimacy_matrix(intimacy_matrix, agents)

    if dampening < 0:
        raise ValueError(f"dampening must be nonnegative, but received {dampening}.")
    if clip_min >= clip_max:
        raise ValueError(f"clip_min must be less than clip_max, but received clip_min={clip_min} and clip_max={clip_max}.")

    leader = agents[leader_index]
    member_indices = get_member_indices(agents, leader_index)

    if "emotion" not in leader:
        raise KeyError("Leader must contain the key 'emotion'.")

    for member_index in member_indices:
        member = agents[member_index]

        required_keys = {"emotion", "delta", "index", "role"}
        missing = required_keys - set(member.keys())
        if missing:
            raise KeyError(f"Member agent at index {member_index} is missing required key(s): {sorted(missing)}.")

        if member["index"] != member_index:
            raise ValueError(f"Member agent at list position {member_index} has index={member['index']}, but expected {member_index}.")

        influence_weight = float(intimacy_matrix[leader_index, member_index])

        member["emotion"] += dampening * (leader["emotion"] - member["emotion"]) * member["delta"] * influence_weight
        member["emotion"] = float(np.clip(member["emotion"], clip_min, clip_max))

    return agents


def run_leader_intervention(
    style: str,
    avg_emotional_valence: float,
    agents: List[dict],
    leader_index: int,
    intimacy_matrix: np.ndarray,
    dampening: float = 0.08,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> Tuple[List[dict], bool]:
    """
    Check whether the leader should intervene and, if so, apply the intervention.

    Returns updated agents and a boolean indicating whether intervention occurred.
    """
    validate_leader_style(style)
    validate_leader_index(agents, leader_index)
    validate_intimacy_matrix(intimacy_matrix, agents)

    leader = agents[leader_index]
    intervened = should_leader_intervene(style=style, avg_emotional_valence=avg_emotional_valence, leader=leader)

    if intervened:
        agents = apply_leader_intervention(
            agents=agents,
            leader_index=leader_index,
            intimacy_matrix=intimacy_matrix,
            dampening=dampening,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    return agents, intervened


def summarize_leader_intervention(style: str, agents: List[dict], leader_index: int, dampening: float) -> Dict[str, object]:
    """
    Create a small summary dictionary describing the current leader setup.
    """
    validate_leader_style(style)
    validate_leader_index(agents, leader_index)

    leader = agents[leader_index]

    return {
        "style": style,
        "leader_index": leader_index,
        "leader_emotion": leader.get("emotion"),
        "emotionManagementAbility": leader.get("emotionManagementAbility"),
        "interventionThreshold": leader.get("interventionThreshold"),
        "dampening": dampening,
    }