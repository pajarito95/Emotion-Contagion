"""
Tabular Q-learning components for the Emotion Contagion ABM.

Purpose:
This file isolates all reinforcement learning logic so RL can be:
    - turned on/off easily
    - swapped independently from network/emotion code
    - imported cleanly into run_simulation.py

Main components:
- RL_ACTIONS
- compute_homophily_index(...)
- compute_state(...)
- compute_quality(...)
- apply_leader_action(...)
- QLearningLeaderPolicy
- leader_behaviors
"""

from __future__ import annotations
import numpy as np

# RL ACTION SPACE: 0 = no intervention; 1 = weak; 2 = medium; 3 = strong
RL_ACTIONS = [0, 1, 2, 3]


# HOMOPHILY METRIC
def compute_homophily_index(agents, intimacy_matrix, tau: float = 0.35):
    """
    Scalar homophily index.

    Interpretation:
        Higher values:
            stronger intimacy among emotionally similar agents

        Lower / negative values:
            stronger intimacy among emotionally dissimilar agents

    Parameters:
        agents : list[dict]
            Agent dictionaries containing emotion values.

        intimacy_matrix : np.ndarray
            NxN weighted directed intimacy matrix.

        tau : float
            Threshold for emotional similarity.

    Returns:
      float
    """
    member_indices = [i for i, agent in enumerate(agents) if agent.get("role") == "member"]
    emos = np.array([agents[i]["emotion"] for i in member_indices], dtype=float)
    N = len(emos)

    W = intimacy_matrix[np.ix_(member_indices, member_indices)].copy()
    np.fill_diagonal(W, 0.0) # ignore self-ties

    similar_weights = []
    dissimilar_weights = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            diff = abs(emos[i] - emos[j])
            w = W[i, j]

            if diff <= tau:
                similar_weights.append(w)
            else:
                dissimilar_weights.append(w)

    if len(similar_weights) == 0:
        return 0.0

    if len(dissimilar_weights) == 0:
        return 0.0

    return float(np.mean(similar_weights) - np.mean(dissimilar_weights))


# RL STATE
def compute_state(agents, intimacy_matrix):
    """
    RL state representation.

    State vector: [mean_emotion, variance_emotion, homophily_index]

    Returns:
        np.ndarray shape (3,)
    """

    emos = np.array([agent["emotion"] for agent in agents if agent.get("role") == "member"], dtype=float)
    mean_emotion = emos.mean()
    variance_emotion = emos.var()
    homophily_index = compute_homophily_index(agents, intimacy_matrix)

    return np.array([mean_emotion, variance_emotion, homophily_index], dtype=float)


# QUALITY FUNCTION
def compute_quality(agents, w1: float = 1.0, w2: float = 0.5):
    """
    Reward quality function. q = mean_emotion - 0.5 * variance

    Higher if:
        - group emotions are positive
        - emotions are more aligned

    Returns:
    float
    """
    emos = np.array([agent["emotion"] for agent in agents if agent.get("role") == "member"], dtype=float)
    mean_emotion = emos.mean()
    variance_emotion = emos.var()

    return (w1 * mean_emotion) - (w2 * variance_emotion)


# LEADER ACTION APPLICATION
def apply_leader_action(action, agents,  leader_intimacy):
    """
    Apply RL leader intervention.

    Actions:
        0 : no intervention
        1 : weak
        2 : medium
        3 : strong

    Parameters:
        action : int
        agents : list[dict]
        leader : dict
        leader_intimacy : np.ndarray
            Leader-to-agent intimacy vector.
    """

    if action == 0:
        return

    if action == 1:
        dampening = 0.02

    elif action == 2:
        dampening = 0.05

    elif action == 3:
        dampening = 0.08

    else:
        raise ValueError(f"Unknown RL action: {action}")

    leader = agents

    for agent in agents:
        delta = agent["delta"]
        agent["emotion"] += (dampening * (leader["emotion"] - agent["emotion"]) * delta * leader_intimacy[agent["index"]])
        agent["emotion"] = np.clip(agent["emotion"], -1, 1)


# TABULAR Q-LEARNING POLICY
class QLearningLeaderPolicy:
    """
    Tabular Q-learning policy.

    State: [mean_emotion, variance_emotion, homophily]
    Action: discrete intervention strength
    Q dimensions: [mean_bin, variance_bin, homophily_bin, action]
    """

    def __init__(
        self,
        actions,
        n_bins_mean: int = 5,
        n_bins_var: int = 4,
        n_bins_homo: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 0.30,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2000,
    ):

        self.actions = list(actions)
        self.nA = len(self.actions)

        self.n_bins_mean = n_bins_mean
        self.n_bins_var = n_bins_var
        self.n_bins_homo = n_bins_homo

        # learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma

        # epsilon-greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.epsilon = epsilon_start

        self.step_count = 0

        # Q table
        self.Q = np.zeros((n_bins_mean, n_bins_var, n_bins_homo, self.nA), dtype=float)

    # discretisation helpers
    def _bin_value(self, value, vmin, vmax, n_bins):
        value_clipped = max(vmin, min(vmax, value))
        frac = ((value_clipped - vmin) / (vmax - vmin + 1e-9))
        idx = int(frac * n_bins)
        if idx == n_bins:
            idx -= 1

        return idx

    def _state_indices(self, state):
        """
        Continuous -> discrete state bins.
        """
        mean_emotion, variance_emotion, homophily = state
        i_mean = self._bin_value(mean_emotion,-1.0, 1.0, self.n_bins_mean)
        i_var = self._bin_value(variance_emotion, 0.0, 1.0, self.n_bins_var)

        # original approximate homophily range
        i_homo = self._bin_value(homophily, 0.0, 0.05, self.n_bins_homo)

        return (i_mean, i_var, i_homo)

    # action selection
    def choose_action(self, state, rng):
        """
        epsilon-greedy action selection
        """
        idx = self._state_indices(state)

        # epsilon decay
        self.step_count += 1
        frac = min(1.0, self.step_count / self.epsilon_decay_steps)
        self.epsilon = (self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start))

        # explore
        if rng.random() < self.epsilon:
            return rng.choice(self.actions)

        # exploit
        q_values = self.Q[idx]
        best_action_idx = int(np.argmax(q_values))

        return self.actions[best_action_idx]

    # Q update
    def update(self, state, action, reward, next_state, done: bool = False):
        """
        Standard tabular Q-learning update.
        """
        s_idx = self._state_indices(state)
        sp_idx = self._state_indices(next_state)
        a_idx = self.actions.index(action)
        q_sa = self.Q[s_idx + (a_idx,)]

        if done:
            target = reward
        else:
            max_q_next = np.max(self.Q[sp_idx])
            target = reward + (self.gamma * max_q_next)

        self.Q[s_idx + (a_idx,)] = (q_sa + self.alpha * (target - q_sa))

# LEADER MODES
leader_behaviors = {
    "No_Intervention": {
        "uses_rl": False,
        "threshold_mode": "never"
    },
    "High_Fully_Constrained": {
        "uses_rl": True,
        "threshold_mode": "always"
    },
    "Low_Fully_Constrained": {
        "uses_rl": True,
        "threshold_mode": "always"
    },
    "High_Initially_Constrained": {
        "uses_rl": True,
        "threshold_mode": "initial"
    },
    "Low_Initially_Constrained": {
        "uses_rl": True,
        "threshold_mode": "initial"
    },
    "Free": {
        "uses_rl": True,
        "threshold_mode": "never"
    }
}