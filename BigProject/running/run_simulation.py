"""
Run one complete emotion contagion simulation.

Current design assumptions:
- All agents, including the leader, are stored in one shared `agents` list.
- The leader is identified by `leader_index`.
- One unified intimacy matrix stores all pairwise ties.
- Member contagion is handled in member_dynamics.py.
- Leader intervention is handled in leader_intervention.py.
- RL is not included here yet.
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


@dataclass
class SimulationResults:
    """
    Container for outputs from one simulation run.
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
    absorption_history: List[Dict[Tuple[int, int], float]]
    final_absorption_dict: Dict[Tuple[int, int], float]
    metadata: Dict[str, Any]
    leader_summary: Dict[str, Any]


def _validate_adaptive_params(
    adaptive_intimacy: bool,
    kappa: Optional[float],
    decay: Optional[float],
    min_w: Optional[float],
    max_w: Optional[float],
) -> None:
    if adaptive_intimacy and any(param is None for param in [kappa, decay, min_w, max_w]):
        raise ValueError("adaptive_intimacy=True requires kappa, decay, min_w, and max_w to all be provided.")


def _snapshot_initial_conditions(agents: List[dict]) -> pd.DataFrame:
    rows = []
    keys = [
        "index",
        "role",
        "emotion",
        "delta",
        "expressiveness",
        "amplification",
        "bias",
        "emotionManagementAbility",
        "interventionThreshold",
    ]

    for agent in agents:
        row = {key: agent.get(key, None) for key in keys}
        rows.append(row)

    return pd.DataFrame(rows)


def _snapshot_emotions(agents: List[dict]) -> List[float]:
    return [float(agent["emotion"]) for agent in agents]


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
    **kwargs,
) -> SimulationResults:
    """
    Run one complete simulation from initialization through the final timestep.
    """
    if not isinstance(max_iterations, int):
        raise TypeError(f"max_iterations must be an integer, but received {type(max_iterations).__name__}.")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, but received {max_iterations}.")

    _validate_adaptive_params(
        adaptive_intimacy=adaptive_intimacy,
        kappa=kappa,
        decay=decay,
        min_w=min_w,
        max_w=max_w,
    )

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

    initial_conditions = _snapshot_initial_conditions(state.agents)

    emotion_history: List[List[float]] = []
    avg_emotion_history: List[float] = []
    buddies_per_timestep: List[List[Tuple[int, int]]] = []
    interactions_per_timestep: List[int] = []
    intervention_timesteps: List[int] = []
    absorption_history: List[Dict[Tuple[int, int], float]] = []
    absorption_dict: Dict[Tuple[int, int], float] = {}

    emotion_history.append(_snapshot_emotions(state.agents))

    time = 0
    while time < max_iterations:
        buddies, absorption_dict = agent_interaction(
            rng=rng,
            agents=state.agents,
            intimacyMatrix=state.intimacy_matrix,
            absorption_dict=absorption_dict,
            leader_index=state.leader_index,
        )

        buddies_per_timestep.append(buddies)
        interactions_per_timestep.append(len(buddies))

        avg_emotional_valence = avgEmotion(
            agents=state.agents,
            leader_index=state.leader_index,
        )
        avg_emotion_history.append(avg_emotional_valence)

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

        emotion_history.append(_snapshot_emotions(state.agents))
        absorption_history.append(dict(absorption_dict))

        time += 1
        state.time = time

    leader_summary = summarize_leader_intervention(
        style=leader_style,
        agents=state.agents,
        leader_index=state.leader_index,
        dampening=dampening,
    )

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
        absorption_history=absorption_history,
        final_absorption_dict=absorption_dict,
        metadata=metadata,
        leader_summary=leader_summary,
    )