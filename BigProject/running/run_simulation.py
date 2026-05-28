"""
Run one complete emotion contagion simulation.

Current design assumptions:
- All agents, including the leader, are stored in one shared `agents` list.
- The leader is identified by `leader_index`.
- One unified intimacy matrix stores all pairwise ties.
- Member contagion is handled in member_dynamics.py.
- Leader intervention is handled through RL or fixed intervention logic.
- RL logic is separated into rl_q.py.
- Plotting and downstream statistical analysis are handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from build_simulation import initialize_simulation
from member_dynamics import avgEmotion, agent_interaction, update_intimacy_matrix
from leader_intervention import run_leader_intervention, summarize_leader_intervention
from rl_q import (
    RL_ACTIONS,
    leader_behaviors,
    compute_state,
    compute_quality,
    apply_leader_action, 
    QLearningLeaderPolicy
    )

@dataclass
class SimulationResults:
    """
    Container for outputs from one simulation run
    """
    run_id: Any
    seed: int
    state: Any
    initial_conditions: pd.DataFrame
    emotion_history: List[List[float]]
    avg_emotion_history: List[float]
    buddies_per_timestep: List[List[Tuple[int, int]]]
    interactions_per_timestep: List[int]
    intervention_timesteps: List[int]
    intervention_log: Dict[int, int]
    absorption_history: List[Dict[Tuple[int, int], float]]
    final_absorption_dict: Dict[Tuple[int, int], float]
    intimacy_matrix_history: List[np.ndarray]
    rl_actions: List[int]
    rl_rewards: List[float]
    rl_quality: List[float]
    homophily_history: List[float]
    metadata: Dict[str, Any]
    leader_summary: Dict[str, Any]
    q_table: Optional[np.ndarray] = None

def _validate_adaptive_params(
    adaptive_intimacy: bool,
    kappa: Optional[float],
    decay: Optional[float],
    min_w: Optional[float],
    max_w: Optional[float],
) -> None:
    if adaptive_intimacy and any(param is None for param in [kappa, decay, min_w, max_w]):
        raise ValueError("adaptive_intimacy=True requires kappa, decay, min_w, and max_w.")

def _snapshot_initial_conditions(agents: List[dict]) -> pd.DataFrame:
    rows = []
    keys = ["index", "role", "emotion", "delta", "expressiveness", "amplification", "bias", "emotionManagementAbility", "interventionThreshold"]
    for agent in agents:
        row = {key: agent.get(key, None) for key in keys}
        rows.append(row)

    return pd.DataFrame(rows)

def _snapshot_emotions(agents: List[dict]) -> List[float]:
    return [float(agent["emotion"]) for agent in agents]

def _compute_homophily_index(
    agents: List[dict],
    intimacy_matrix: np.ndarray,
    leader_index: int,
    tau: float = 0.35,
) -> float:
    """
    Scalar homophily metric. Higher value => stronger weighting toward emotionally similar agents.
    """
    member_indices = [i for i, a in enumerate(agents) if i != leader_index and a.get("role") == "member"]
    emos = np.array([agents[i]["emotion"] for i in member_indices])
    if len(emos) < 2:
        return 0.0

    W = intimacy_matrix[np.ix_(member_indices, member_indices)].copy()
    np.fill_diagonal(W, 0.0)

    similar_weights = []
    dissimilar_weights = []

    for i in range(len(member_indices)):
        for j in range(len(member_indices)):
            if i == j:
                continue

            diff = abs(emos[i] - emos[j])
            w = W[i, j]

            if diff <= tau:
                similar_weights.append(w)
            else:
                dissimilar_weights.append(w)

    # TODO: Consider different apprach
    if len(similar_weights) == 0 or len(dissimilar_weights) == 0:
        return 0.0

    return float(np.mean(similar_weights) - np.mean(dissimilar_weights))

def run_simulation(
    seed: int,
    run_id: Any,
    population_size: int,
    structure: str,
    leader_style: str,
    max_iterations: int,
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
    adaptive_intimacy: bool = False,
    kappa: Optional[float] = None,
    decay: Optional[float] = None,
    min_w: Optional[float] = None,
    max_w: Optional[float] = None,
    dampening: float = 0.08,
    condition_name: Optional[str] = None,
    use_rl_leader: bool = False,
    rl_alpha: float = 0.1,
    rl_gamma: float = 0.95,
    rl_epsilon_start: float = 0.3,
    rl_epsilon_end: float = 0.05,
    rl_epsilon_decay_steps: int = 2000,
    **kwargs,
) -> SimulationResults:
    """
    Run one complete simulation.
    """
    if not isinstance(max_iterations, int):
        raise TypeError(f"max_iterations must be an integer, received {type(max_iterations).__name__}.")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, received {max_iterations}.")

    _validate_adaptive_params(adaptive_intimacy=adaptive_intimacy, kappa=kappa, decay=decay, min_w=min_w, max_w=max_w)

    rng = np.random.default_rng(seed)
    state = initialize_simulation(
        rng=rng,
        population_size=population_size,
        structure=structure,
        leader_style=leader_style,
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

    behavior = leader_behaviors[leader_style]
    if use_rl_leader and behavior["uses_rl"]:
        policy = QLearningLeaderPolicy(
            actions=RL_ACTIONS,
            alpha=rl_alpha,
            gamma=rl_gamma,
            epsilon_start=rl_epsilon_start,
            epsilon_end=rl_epsilon_end,
            epsilon_decay_steps=rl_epsilon_decay_steps,
        )
    else:
        policy = None

    initial_conditions = _snapshot_initial_conditions(state.agents)

    emotion_history: List[List[float]] = []
    avg_emotion_history: List[float] = []
    buddies_per_timestep: List[List[Tuple[int, int]]] = []
    interactions_per_timestep: List[int] = []
    intervention_timesteps: List[int] = []
    intervention_log: Dict[int, int] = {}
    absorption_history: List[Dict[Tuple[int, int], float]] = []
    absorption_dict: Dict[Tuple[int, int], float] = {}
    intimacy_matrix_history: List[np.ndarray] = []
    rl_actions: List[int] = []
    rl_rewards: List[float] = []
    rl_quality: List[float] = []
    homophily_history: List[float] = []

    emotion_history.append(_snapshot_emotions(state.agents))
    initial_homophily = _compute_homophily_index( agents=state.agents, intimacy_matrix=state.intimacy_matrix, leader_index=state.leader_index)
    homophily_history.append(initial_homophily)

    if policy is not None:
        state_t = compute_state(agents=state.agents, intimacy_matrix=state.intimacy_matrix, leader_index=state.leader_index, homophily_value=initial_homophily)
        prev_quality = compute_quality( agents=state.agents, leader_index=state.leader_index)
        rl_quality.append(prev_quality)

    threshold_mode = behavior["threshold_mode"]
    leader_intervened = False

    time = 0
    intimacy_matrix_history.append(state.intimacy_matrix.copy())
    while time < max_iterations:
        done = (time == max_iterations - 1)

        # RL ACTION SELECTION
        action_t = 0

        if policy is not None:
            avg_emotional_valence = avgEmotion(agents=state.agents, leader_index=state.leader_index)

            leader = state.agents[state.leader_index]
            threshold = leader.get("interventionThreshold")

            if threshold_mode == "never":
                action_t = policy.choose_action(state_t, rng)
            elif threshold_mode == "always":
                if (threshold is not None and avg_emotional_valence <= threshold):
                    action_t = policy.choose_action(state_t, rng)
            elif threshold_mode == "initial":
                if not leader_intervened:
                    if (threshold is not None and avg_emotional_valence <= threshold):
                        action_t = policy.choose_action(state_t, rng)
                else:
                    action_t = policy.choose_action(state_t, rng)

        rl_actions.append(action_t)

        # MEMBER INTERACTIONS
        buddies, absorption_dict = agent_interaction(rng=rng, agents=state.agents, intimacyMatrix=state.intimacy_matrix, absorption_dict=absorption_dict, leader_index=state.leader_index)
        buddies_per_timestep.append(buddies)
        interactions_per_timestep.append(len(buddies))

        avg_emotional_valence = avgEmotion(agents=state.agents, leader_index=state.leader_index)
        avg_emotion_history.append(avg_emotional_valence)

        # LEADER INTERVENTION
        if policy is not None:
            apply_leader_action(action=action_t, agents=state.agents, leader_index=state.leader_index, intimacy_matrix=state.intimacy_matrix)

            if action_t != 0:
                intervention_timesteps.append(time)
                intervention_log[time] = action_t
                leader_intervened = True

        else:
            state.agents, intervened = run_leader_intervention(
                style=leader_style,
                avg_emotional_valence=avg_emotional_valence,
                agents=state.agents,
                leader_index=state.leader_index,
                intimacy_matrix=state.intimacy_matrix,
                dampening=dampening,
            )
            if intervened:
                intervention_timesteps.append(time)
                intervention_log[time] = 1

        # ADAPTIVE INTIMACY
        if adaptive_intimacy:
            state.intimacy_matrix = update_intimacy_matrix(
                intimacy=state.intimacy_matrix,
                agents=state.agents,
                leader_index=state.leader_index,
                kappa=kappa,
                decay=decay,
                min_w=min_w,
                max_w=max_w,
            )
            intimacy_matrix_history.append(state.intimacy_matrix.copy())

        # HOMOPHILY
        homophily_t = _compute_homophily_index(agents=state.agents, intimacy_matrix=state.intimacy_matrix, leader_index=state.leader_index)
        homophily_history.append(homophily_t)
 
        # RL UPDATE
        if policy is not None:
            new_quality = compute_quality(agents=state.agents, leader_index=state.leader_index)
            reward_t = new_quality - prev_quality
            prev_quality = new_quality

            state_tp1 = compute_state(agents=state.agents, intimacy_matrix=state.intimacy_matrix, leader_index=state.leader_index, homophily_value=homophily_t)

            policy.update(state=state_t, action=action_t, reward=reward_t, next_state=state_tp1, done=done)

            rl_rewards.append(reward_t)
            rl_quality.append(new_quality)

            state_t = state_tp1

        # LOGGING
        emotion_history.append(_snapshot_emotions(state.agents))

        absorption_history.append(dict(absorption_dict))

        time += 1
        state.time = time

    leader_summary = summarize_leader_intervention(style=leader_style, agents=state.agents, leader_index=state.leader_index, dampening=dampening)

    metadata = dict(state.metadata)
    metadata.update(
        {
            "condition_name": condition_name,
            "max_iterations": max_iterations,
            "adaptive_intimacy": adaptive_intimacy,
            "kappa": kappa,
            "decay": decay,
            "min_w": min_w,
            "max_w": max_w,
            "use_rl_leader": use_rl_leader,
        }
    )

    return SimulationResults(
        run_id=run_id,
        seed=seed,
        state=state,
        initial_conditions=initial_conditions,
        emotion_history=emotion_history,
        avg_emotion_history=avg_emotion_history,
        buddies_per_timestep=buddies_per_timestep,
        interactions_per_timestep=interactions_per_timestep,
        intervention_timesteps=intervention_timesteps,
        intervention_log=intervention_log,
        absorption_history=absorption_history,
        final_absorption_dict=absorption_dict,
        intimacy_matrix_history=intimacy_matrix_history,
        rl_actions=rl_actions,
        rl_rewards=rl_rewards,
        rl_quality=rl_quality,
        homophily_history=homophily_history,
        metadata=metadata,
        leader_summary=leader_summary,
        q_table=policy.Q.copy() if policy is not None else None,
    )