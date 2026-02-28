from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class SimulationState:
    agents: list
    leader: object
    intimacy_matrix: np.ndarray
    leader_intimacy: np.ndarray
    assignments: np.ndarray