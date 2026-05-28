"""
Dataclasses for storing the state of the emotion contagion simulation.

Current design assumptions:
- All agents, leader included, are stored in one shared `agents` list.
- Leader is identified by `leader_index`.
- One unified intimacy matrix stores all (pairwise) ties.
- Agents have a role field as "leader" or "member".
- Emotional contagion updates are only applied to members (for now at least).
- Leader intervention is handled separately in leader_intervention.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SimulationState:
    """
    Container for the current state of one simulation.

    Parameters:
        agents: list[dict]
            All agents in the simulation, leader included
            Each agent is currently represented as a dictionary
        leader_index: int
            Index of leader within `agents`
        intimacy_matrix: np.ndarray
            Complete NxN intimacy matrix over all agents
        assignments: np.ndarray or None, optional
            Group (community, core-periphery, random) assignments used to generate the network
            May be None if chosen structure does not use assignments
        time: int, optional
            Current simulation timestep
        metadata: dict, optional
            Optional dictionary for storing setup/configuration details that are useful to keep alongside the state
    """
    agents: List[Dict[str, Any]]
    leader_index: int
    intimacy_matrix: np.ndarray
    assignments: Optional[np.ndarray] = None
    time: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate the internal consistency of the state after initialization.
        """
        if not isinstance(self.agents, list):
            raise TypeError(f"agents must be a list, but received {type(self.agents).__name__}.")

        if len(self.agents) == 0:
            raise ValueError("agents cannot be empty.")

        if not isinstance(self.leader_index, int):
            raise TypeError(f"leader_index must be an integer, but received {type(self.leader_index).__name__}.")

        if not (0 <= self.leader_index < len(self.agents)):
            raise ValueError(f"leader_index={self.leader_index} is out of bounds for {len(self.agents)} agents.")

        if not isinstance(self.intimacy_matrix, np.ndarray):
            raise TypeError("intimacy_matrix must be a NumPy array.")

        if self.intimacy_matrix.ndim != 2:
            raise ValueError(f"intimacy_matrix must be 2D, but received shape {self.intimacy_matrix.shape}.")

        n_agents = len(self.agents)
        expected_shape = (n_agents, n_agents)
        if self.intimacy_matrix.shape != expected_shape:
            raise ValueError(f"intimacy_matrix shape {self.intimacy_matrix.shape} does not match the required shape {expected_shape} for {n_agents} agents.")

        leader = self.agents[self.leader_index]
        role = leader.get("role", None)
        if role != "leader":
            raise ValueError(f"Agent at leader_index={self.leader_index} must have role='leader', but has role={role!r}.")

        for i, agent in enumerate(self.agents):
            if not isinstance(agent, dict):
                raise TypeError(f"agents[{i}] must be a dictionary, but received {type(agent).__name__}.")

            if "index" not in agent:
                raise KeyError(f"agents[{i}] is missing the required key 'index'.")

            if agent["index"] != i:
                raise ValueError(f"agents[{i}]['index'] must equal its position in the list. Expected {i}, received {agent['index']}.")

            if "role" not in agent:
                raise KeyError(f"agents[{i}] is missing the required key 'role'.")

            if agent["role"] not in {"leader", "member"}:
                raise ValueError(f"agents[{i}]['role'] must be either 'leader' or 'member', but received {agent['role']!r}.")

        n_leaders = sum(agent["role"] == "leader" for agent in self.agents)
        if n_leaders != 1:
            raise ValueError(f"Exactly one leader is required, but found {n_leaders}.")

    @property
    def leader(self) -> Dict[str, Any]:
        """
        Return leader dictionary directly
        """
        return self.agents[self.leader_index]

    @property
    def members(self) -> List[Dict[str, Any]]:
        """
        Return non-leader agents
        """
        return [agent for i, agent in enumerate(self.agents) if i != self.leader_index]

    @property
    def member_indices(self) -> List[int]:
        """
        Return indices of non-leader agents
        """
        return [i for i in range(len(self.agents)) if i != self.leader_index]

    @property
    def n_agents(self) -> int:
        """
        Total number of agents including leader
        """
        return len(self.agents)

    @property
    def n_members(self) -> int:
        """
        Number of non-leader agents
        """
        return len(self.agents) - 1