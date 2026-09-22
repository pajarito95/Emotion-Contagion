"""
Follower/member emotional contagion and adaptive intimacy logic for the emotion contagion ABM.

Current design assumptions:
- All agents, including the leader, are stored in one shared `agents` list.
- Leader is identified by the last element in agents list.
- One intimacy matrix storing member ties.
- Regular emotional contagion is applied only to members.
- Leader intervention is handled separately in leader_intervention.py.
- Adaptive intimacy updates here affect member-member ties only.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# def get_member_indices(agents: List[dict], leader_index: int) -> List[int]:
#     """
#     Return the indices of all non-leader agents.
#     Parameters:
#         agents: list[dict]
#             Full agent list including the leader
#         leader_index: int
#             Index of the leader
#     Returns:
#         list[int]
#             Indices of member agents only.
#     """
#     if not isinstance(agents, list) or len(agents) == 0:
#         raise ValueError("agents must be a non-empty list.")
#     if not isinstance(leader_index, int):
#         raise TypeError("leader_index must be an integer.")
#     if not (0 <= leader_index < len(agents)):
#         raise ValueError(f"leader_index={leader_index} is out of bounds for {len(agents)} agents.")
#     return [i for i, agent in enumerate(agents) if agent.get("role") == "member"]


def avgEmotion(agents: List[dict]) -> float:
    """
    Calculate average emotional valence across members.

    Parameters:
        agents: list[dict]
            Full agent list
        
    Returns:
        float
            Mean emotional valence of the members
    """
    if len(agents) == 0:
        raise ValueError("avgEmotion received an empty agents list.")

    if agents[-1].get("role") != "leader":
        raise ValueError("Last agent expected to be leader.")

    member_emotions = [agent["emotion"] for agent in agents[:-1]]

    if len(member_emotions) == 0:
        raise ValueError("No member agents were found when computing avgEmotion.")

    return float(sum(member_emotions) / len(member_emotions))


def update_intimacy_matrix(
    intimacy: np.ndarray,
    include_leader_ties: bool,
    agents: List[dict],
    kappa: float,
    decay: float,
    min_w: float,
    max_w: float,
) -> np.ndarray:
    """
    Adapt member-member intimacy weights based on emotional similarity.

    Current behavior:
    - member-member ties are updated here
    - self-ties set to zero
    - each updated member row is renormalized

    Parameters:
        intimacy: np.ndarray
            N-1xN-1 intimacy matrix over member agents
        agents: list[dict]
            Full agent list
        kappa: float
            Strength of similarity-based gain
        decay: float
            Forgetting/decay factor applied to member-member ties
        min_w: float
            Minimum post-update member-member weight
        max_w: float
            Maximum post-update member-member weight

    Returns:
        np.ndarray
            Updated intimacy matrix
    """
    # ── Determine matrix size ──
    # If leader ties are included, the matrix is N×N (members + leader);
    # otherwise it's (N-1)×(N-1) (members only).
    n_agents = (len(agents) if include_leader_ties else len(agents) - 1)

    # ── Input validation ──
    if n_agents == 0:
        raise ValueError("update_intimacy_matrix() received an empty agents list.")
    if agents[-1].get("role") != "leader":
        raise ValueError("Expected last agent to be the leader.")
    if intimacy.ndim != 2 or intimacy.shape[0] != intimacy.shape[1]:
        raise ValueError("update_intimacy_matrix() expected a square 2D intimacy matrix.")
    if intimacy.shape[0] != n_agents:
        raise ValueError(f"Intimacy matrix size ({intimacy.shape[0]}) does not match number of agents ({n_agents}).")
    if not (0 <= decay <= 1):
        raise ValueError("decay must be between 0 and 1.")
    if kappa < 0:
        raise ValueError("kappa must be nonnegative.")
    if not (-1.0 <= min_w <= max_w <= 1.0):
        raise ValueError("Require -1 <= min_w <= max_w <= 1.")

    # ── Copy the matrix so we don't mutate the caller's array ──
    A = intimacy.copy()

    # Extract member emotions as a 1-D array (exclude the leader, who is last).
    # Each entry is the current emotional valence ∈ [-1, 1] of a member.
    emos = np.array([agent["emotion"] for agent in agents[:-1]], dtype=float)

    # ── Compute the pairwise emotional difference matrix ──
    # diff[i, j] = |emotion_i - emotion_j|, ranging from 0 (identical) to 2 (opposite).
    # emos[:, None] is a column vector, emos[None, :] is a row vector;
    # broadcasting their difference gives the full N_members × N_members distance matrix.
    diff = np.abs(emos[:, None] - emos[None, :])

    # ── Compute the homophily gain matrix ──
    # gain[i, j] = κ * (1 - |e_i - e_j|)
    #   - Similar emotions (|diff| ≈ 0) → gain ≈ +κ  (tie gets a positive boost)
    #   - Moderately different (|diff| ≈ 1) → gain ≈ 0  (neutral, no contribution)
    #   - Very different (|diff| > 1) → gain < 0  (tie is penalised)
    #   - Opposite emotions (|diff| = 2) → gain = -κ  (maximum penalty)
    gain = kappa * (1.0 - diff)

    # ── Extract the member-member sub-block from the full matrix ──
    # n_members is the number of non-leader agents.
    # np.ix_ creates an open mesh index that selects the rows AND columns for members only,
    # leaving any leader row/column untouched in A.
    n_members = len(agents) - 1
    member_block = A[np.ix_(range(n_members), range(n_members))]

    # ── Mask the gain to only affect existing (non-zero) ties ──
    # Build a 0/1 mask: 1 where the old weight is non-zero, 0 where it's zero.
    # This prevents gain from creating new ties between agents who currently
    # have no connection — only existing ties can be strengthened or weakened.
    existing_tie_mask = (member_block != 0).astype(float)

    # ── Apply the decay + gain update (existing ties only) ──
    # new_w[i,j] = (1 - decay) * old_w[i,j] + gain[i,j] * mask[i,j]
    # The (1 - decay) factor shrinks the old weight slightly (forgetting),
    # then the gain adds or subtracts based on current emotional similarity —
    # but only for pairs that already have a tie.
    # Zero entries stay zero: (1-decay)*0 + gain*0 = 0.
    member_block = (1.0 - decay) * member_block + gain * existing_tie_mask

    # ── Zero out self-ties (diagonal) ──
    # An agent has no tie to itself, so w[i,i] must be 0.
    np.fill_diagonal(member_block, 0.0)

    # ── Clamp weights to [min_w, max_w] ──
    # Prevents any single tie from growing unboundedly.
    # With min_w = -1.0, negative weights are preserved (hostile ties),
    # but values are still bounded within [-1, 1].
    member_block = np.clip(member_block, min_w, max_w)

    # ── Zero out the diagonal again ──
    # Clamping may have set diagonal entries to min_w, so we re-zero them.
    np.fill_diagonal(member_block, 0.0)

    # ── Compute absolute row sums for normalisation ──
    # Uses np.abs() to match network.py's _normalize_rows: the absolute values
    # in each row must sum to 1 after normalisation, preserving signed weights.
    # This allows negative ties (hostile/repulsive) to coexist with positive ones
    # within the same row, with |w_ij| representing tie strength and sign representing valence.
    # keepdims=True keeps the shape (n_members, 1) so broadcasting works below.
    row_sums = np.abs(member_block).sum(axis=1, keepdims=True)

    # Handle rows where all ties are exactly zero.
    # Instead of raising an error, skip normalisation for those rows by using
    # 1.0 as the divisor — this preserves the zero values as-is, leaving the
    # member effectively isolated (all ties = 0). Normal rows are divided by
    # their abs sum as usual.
    zero_rows = row_sums == 0.0
    safe_sums = np.where(zero_rows, 1.0, row_sums)

    # ── Row-normalise and write back into the full matrix ──
    # Divide each row by its safe sum so |w_ij| values sum to 1 per row
    # (for rows with meaningful content). All-zero rows keep their zero
    # values unchanged. Only the member-member block is overwritten;
    # leader row/column (if present) is untouched.
    A[np.ix_(range(n_members), range(n_members))] = member_block / safe_sums

    return A


def emotion_update(
    agentA: dict,
    agentB: dict,
    agentA_index: int,
    agentB_index: int,
    agents: List[dict],
    intimacyMatrix: np.ndarray,
    absorption_dict: Dict[Tuple[int, int], float]
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
            N-1xN-1 intimacy matrix over member agents
        absorption_dict: dict
            Dictionary storing cumulative absolute emotional changes by ordered pair

    Returns:
        dict
            Updated absorption dictionary
    """
    if agentA.get("role") != "member" or agentB.get("role") != "member":
        raise ValueError("emotion_update() expects both interacting agents to have role='member'.")

    if (agentB_index, agentA_index) not in absorption_dict:
        absorption_dict[(agentB_index, agentA_index)] = 0.0

    if (agentA_index, agentB_index) not in absorption_dict:
        absorption_dict[(agentA_index, agentB_index)] = 0.0

    initial_qA = agentA["emotion"]
    initial_qB = agentB["emotion"]

    gamma_A = sum(sender["expressiveness"] * intimacyMatrix[sender["index"], agentA["index"]] * agentA["susceptibility"] for sender in agents[:-1] if sender is not agentA)
    gamma_B = sum(sender["expressiveness"] * intimacyMatrix[sender["index"], agentB["index"]] * agentB["susceptibility"] for sender in agents[:-1] if sender is not agentB)

    eta_A = agentA["amplification"]
    eta_B = agentB["amplification"]
    beta_A = agentA["bias"]
    beta_B = agentB["bias"]

    # OLD: Weighted average of other members' emotions
    # groupEmos_A = sum(other["expressiveness"] * intimacyMatrix[other["index"], agentA["index"]] for other in agents[:-1] if other is not agentA)
    # groupEmos_B = sum(other["expressiveness"] * intimacyMatrix[other["index"], agentB["index"]] for other in agents[:-1] if other is not agentB)

    # if groupEmos_A == 0 or groupEmos_B == 0:
    #     raise ValueError("Encountered zero weighted expressiveness while computing q*. Check member-member intimacy normalization and expressiveness values.")
 
    # qstar_A = sum(((sender["expressiveness"]  * intimacyMatrix[sender["index"], agentA["index"]]) / groupEmos_A)  * sender["emotion"] for sender in agents[:-1] if sender is not agentA)
    # qstar_B = sum(((sender["expressiveness"] * intimacyMatrix[sender["index"], agentB["index"]]) / groupEmos_B) * sender["emotion"] for sender in agents[:-1] if sender is not agentB)

    # NEW: Plain unweighted average of other members' emotions, excluding self, e_{N(i)}
    qstar_A = sum(sender["emotion"] for sender in agents[:-1] if sender is not agentA) / (len(agents[:-1]) - 1)  # minus 1 because for the self exclusion
    qstar_B = sum(sender["emotion"] for sender in agents[:-1] if sender is not agentB) / (len(agents[:-1]) - 1)

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
    include_leader_ties: bool
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
            N-1xN-1 intimacy matrix
        absorption_dict: dict
            Cumulative absorption/change tracker

    Returns:
        tuple[list[tuple[int, int]], dict]
            Interacting member index pairs in full-agent indexing and the updated absorption dictionary.
    """
    n_agents = (len(agents) if include_leader_ties else len(agents)-1)
    if agents[-1].get("role") != "leader":
        raise ValueError("Expected last agent to be the leader.")
    if intimacyMatrix.shape != (n_agents, n_agents):
        raise ValueError(f"Intimacy matrix shape {intimacyMatrix.shape} does not match {n_agents} agents.")

    buddies: list[tuple[int, int]] = []

    buddies_agents = len(agents[:-1])
    for pos_a, i in enumerate(range(buddies_agents)):
        for j in range(pos_a + 1, buddies_agents):
            interaction_prob = max(0.0, intimacyMatrix[i, j], intimacyMatrix[j, i])
            if rng.random() < interaction_prob:
                buddies.append((i, j))

    for i, j in buddies:
        agentA, agentB = agents[i], agents[j]
        absorption_dict = emotion_update(
            agentA=agentA,
            agentB=agentB,
            agentA_index=i,
            agentB_index=j,
            agents=agents,
            intimacyMatrix=intimacyMatrix,
            absorption_dict=absorption_dict
        )

    return buddies, absorption_dict