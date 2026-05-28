"""
Follower/member emotional contagion and adaptive intimacy logic for the emotion contagion ABM.

Current design assumptions:
- All agents, including the leader, are stored in one shared `agents` list.
- Leader is identified by `leader_index`.
- One unified intimacy matrix stores all pairwise ties.
- Regular emotional contagion is applied only to members.
- Leader intervention is handled separately in leader_intervention.py.
- Adaptive intimacy updates here affect member-member ties only.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def get_member_indices(agents: List[dict], leader_index: int) -> List[int]:
    """
    Return the indices of all non-leader agents.

    Parameters:
        agents: list[dict]
            Full agent list including the leader
        leader_index: int
            Index of the leader

    Returns:
        list[int]
            Indices of member agents only.
    """
    if not isinstance(agents, list) or len(agents) == 0:
        raise ValueError("agents must be a non-empty list.")
    
    if not isinstance(leader_index, int):
        raise TypeError("leader_index must be an integer.")
    
    if not (0 <= leader_index < len(agents)):
        raise ValueError(f"leader_index={leader_index} is out of bounds for {len(agents)} agents.")

    return [i for i, agent in enumerate(agents) if agent.get("role") == "member"]


def avgEmotion(agents: List[dict], leader_index: int | None = None) -> float:
    """
    Calculate average emotional valence across members.

    Parameters:
        agents: list[dict]
            Full agent list
        leader_index: int or None, optional
            Leader index. If provided, the leader is excluded explicitly. If omitted, all agents with role='member' are used.

    Returns:
        float
            Mean emotional valence of the members
    """
    if len(agents) == 0:
        raise ValueError("avgEmotion received an empty agents list.")

    if leader_index is not None:
        member_indices = get_member_indices(agents, leader_index)
        member_emotions = [agents[i]["emotion"] for i in member_indices]
    else:
        member_emotions = [agent["emotion"] for agent in agents if agent.get("role") == "member"]

    if len(member_emotions) == 0:
        raise ValueError("No member agents were found when computing avgEmotion.")

    return float(sum(member_emotions) / len(member_emotions))


def update_intimacy_matrix(
    intimacy: np.ndarray,
    agents: List[dict],
    leader_index: int,
    kappa: float,
    decay: float,
    min_w: float,
    max_w: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Adapt member-member intimacy weights based on emotional similarity.

    Current behavior:
    - only member-member ties are updated here
    - leader-related ties are preserved as they currently are
    - self-ties are set to zero
    - each updated member row is renormalized across member columns only

    Parameters:
        intimacy: np.ndarray
            Full NxN intimacy matrix over all agents
        agents: list[dict]
            Full agent list including the leader
        leader_index: int
            Index of the leader
        kappa: float
            Strength of similarity-based gain
        decay: float
            Forgetting/decay factor applied to member-member ties
        min_w: float
            Minimum post-update member-member weight
        max_w: float
            Maximum post-update member-member weight
        eps: float, optional
            Small constant to avoid division by zero

    Returns:
        np.ndarray
            Updated intimacy matrix
    """
    n_agents = len(agents)

    if intimacy.ndim != 2 or intimacy.shape[0] != intimacy.shape[1]:
        raise ValueError("update_intimacy_matrix expected a square 2D intimacy matrix.")
    if intimacy.shape[0] != n_agents:
        raise ValueError(f"Intimacy matrix size ({intimacy.shape[0]}) does not match number of agents ({n_agents}).")
    if not (0 <= decay <= 1):
        raise ValueError("decay must be between 0 and 1.")
    if kappa < 0:
        raise ValueError("kappa must be nonnegative.")
    if min_w < 0 or max_w < 0 or min_w > max_w:
        raise ValueError("Require 0 <= min_w <= max_w.")

    member_indices = get_member_indices(agents, leader_index)
    if len(member_indices) == 0:
        raise ValueError("No members found for member-member intimacy updates.")

    A = intimacy.copy()
    emos = np.array([agents[i]["emotion"] for i in member_indices], dtype=float)

    diff = np.abs(emos[:, None] - emos[None,:])
    gain = kappa * (1.0 - diff)

    # Update only member-member block
    member_block = A[np.ix_(member_indices, member_indices)]
    member_block = (1.0 - decay) * member_block + gain

    np.fill_diagonal(member_block, 0.0)
    member_block = np.clip(member_block, min_w, max_w)
    np.fill_diagonal(member_block, 0.0)

    row_sums = member_block.sum(axis=1, keepdims=True)
    if np.any(row_sums <= eps):
        raise ValueError("At least one member intimacy row has near-zero sum after update. Try increasing max_w, decreasing decay, or decreasing min_w.")

    A[np.ix_(member_indices, member_indices)] = member_block / (row_sums + eps)

    return A


def emotional_valence_update(
    agentA: dict,
    agentB: dict,
    agentA_index: int,
    agentB_index: int,
    agents: List[dict],
    intimacyMatrix: np.ndarray,
    absorption_dict: Dict[Tuple[int, int], float],
    leader_index: int,
) -> Dict[Tuple[int, int], float]:
    """
    Update the emotional valence of two interacting member agents according to the Bosse-style contagion rule. Only member-member interactions considered.

    Parameters:
        agentA, agentB: dict
            The two interacting member agents
        agentA_index, agentB_index: int
            Their indices in the full agents list
        agents: list[dict]
            Full agent list including the leader
        intimacyMatrix: np.ndarray
            Full NxN intimacy matrix
        absorption_dict: dict
            Dictionary storing cumulative absolute emotional changes by ordered pair
        leader_index: int
            Index of the leader

    Returns:
        dict
            Updated absorption dictionary
    """
    if agentA_index == leader_index or agentB_index == leader_index:
        raise ValueError("emotional_valence_update received the leader, but only member-member interactions are allowed here.")

    if agentA.get("role") != "member" or agentB.get("role") != "member":
        raise ValueError("emotional_valence_update expects both interacting agents to have role='member'.")

    if (agentB_index, agentA_index) not in absorption_dict:
        absorption_dict[(agentB_index, agentA_index)] = 0.0

    if (agentA_index, agentB_index) not in absorption_dict:
        absorption_dict[(agentA_index, agentB_index)] = 0.0

    initial_qA = agentA["emotion"]
    initial_qB = agentB["emotion"]

    member_indices = get_member_indices(agents, leader_index)
    member_agents = [agents[i] for i in member_indices]

    gamma_A = sum(sender["expressiveness"] * intimacyMatrix[sender["index"], agentA["index"]] * agentA["delta"] for sender in member_agents if sender is not agentA)
    gamma_B = sum(sender["expressiveness"] * intimacyMatrix[sender["index"], agentB["index"]] * agentB["delta"] for sender in member_agents if sender is not agentB)

    eta_A = agentA["amplification"]
    eta_B = agentB["amplification"]
    beta_A = agentA["bias"]
    beta_B = agentB["bias"]

    groupEmos_A = sum(other["expressiveness"] * intimacyMatrix[other["index"], agentA["index"]] for other in member_agents if other is not agentA)
    groupEmos_B = sum(other["expressiveness"] * intimacyMatrix[other["index"], agentB["index"]] for other in member_agents if other is not agentB)

    if groupEmos_A == 0 or groupEmos_B == 0:
        raise ValueError("Encountered zero weighted expressiveness while computing q*. Check member-member intimacy normalization and expressiveness values.")

    qstar_A = sum(((sender["expressiveness"]  * intimacyMatrix[sender["index"], agentA["index"]]) / groupEmos_A)  * sender["emotion"] for sender in member_agents if sender is not agentA)

    qstar_B = sum(((sender["expressiveness"] * intimacyMatrix[sender["index"], agentB["index"]]) / groupEmos_B) * sender["emotion"] for sender in member_agents if sender is not agentB)

    PI_A = 1 - (1 - qstar_A) * (1 - initial_qA)
    NI_A = qstar_A * initial_qA
    PI_B = 1 - (1 - qstar_B) * (1 - initial_qB)
    NI_B = qstar_B * initial_qB

    agentA["emotion"] += gamma_A * (eta_A  * (beta_A * PI_A + (1 - beta_A) * NI_A) + (1 - eta_A) * qstar_A  - initial_qA)
    agentA["emotion"] = float(np.clip(agentA["emotion"], -1.0, 1.0))
    absorption_dict[(agentB_index, agentA_index)] += abs(initial_qA - agentA["emotion"])

    agentB["emotion"] += gamma_B * (eta_B * (beta_B * PI_B + (1 - beta_B) * NI_B) + (1 - eta_B) * qstar_B - initial_qB)
    agentB["emotion"] = float(np.clip(agentB["emotion"], -1.0, 1.0))
    absorption_dict[(agentA_index, agentB_index)] += abs(initial_qB - agentB["emotion"])

    return absorption_dict


def agent_interaction(
    rng: np.random.Generator,
    agents: List[dict],
    intimacyMatrix: np.ndarray,
    absorption_dict: Dict[Tuple[int, int], float],
    leader_index: int,
) -> tuple[list[tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Define pairwise member-member interactions based on intimacy probabilities, then apply the emotional contagion update to each selected pair.

    Current behavior:
    - only non-leader members are eligible to interact here
    - each unordered pair can interact at most once per timestep
    - interaction probability uses the stronger of the two directed ties

    Parameters:
        rng: numpy.random.Generator
            Random number generator
        agents: list[dict]
            Full agent list including the leader
        intimacyMatrix: np.ndarray
            Full NxN intimacy matrix
        absorption_dict: dict
            Cumulative absorption/change tracker
        leader_index: int
            Index of the leader

    Returns:
        tuple[list[tuple[int, int]], dict]
            Interacting member index pairs in full-agent indexing and the updated absorption dictionary.
    """
    n_agents = len(agents)

    if intimacyMatrix.shape != (n_agents, n_agents):
        raise ValueError(f"Intimacy matrix shape {intimacyMatrix.shape} does not match {n_agents} agents.")

    member_indices = get_member_indices(agents, leader_index)
    buddies: list[tuple[int, int]] = []

    for pos_a, i in enumerate(member_indices):
        for j in member_indices[pos_a + 1:]:
            interaction_prob = max(intimacyMatrix[i, j], intimacyMatrix[j, i])

            if rng.random() < interaction_prob:
                buddies.append((i, j))

    for i, j in buddies:
        agentA, agentB = agents[i], agents[j]
        absorption_dict = emotional_valence_update(
            agentA=agentA,
            agentB=agentB,
            agentA_index=i,
            agentB_index=j,
            agents=agents,
            intimacyMatrix=intimacyMatrix,
            absorption_dict=absorption_dict,
            leader_index=leader_index,
        )

    return buddies, absorption_dict