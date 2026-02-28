import numpy as np
from dataclasses import dataclass

@dataclass
class Agent:
    emotion: float
    delta: float
    expressiveness: float
    amplification: float
    bias: float
    index: int

@ dataclass
class Leader: 
    emotion: float
    emotion_management_ability: str
    intervention_threshold: float


def make_agents(rng, population_size):
    '''
    Create a set of agents with certain attributes
    '''
    agents = []
    for _ in range(population_size):
        agent = {
            'emotion': -0.5 + 1 * rng.beta(2, 5),
            'delta': rng.uniform(0, 1),
            'expressiveness': rng.uniform(0, 1),
            'amplification': rng.uniform(0, 1),
            'bias': rng.uniform(0, 1)
        }
        agents.append(agent)

    return agents


def select_leader(rng, agents):
    '''
    Randomly select a leader from the agents and remove them from the team.
    Leader configuration will be done elsewhere
    '''
    leader = rng.choice(agents)
    agents.remove(leader)
    leader_object = Leader(emotion=1.0)

    return leader_object, agents


def configure_leader(leader, style):
    '''
    Configure the leader's emotion management ability and intervention threshold based on the style.
    '''
    if style == 'High_Fully_Constrained' or style == 'High_Initially_Constrained':
        leader.emotion_management_ability = 'High'
        leader.intervention_threshold = -0.5

    elif style == 'Low_Fully_Constrained' or style == 'Low_Initially_Constrained':
        leader.emotion_management_ability = 'Low'
        leader.intervention_threshold = -0.7

    else:
        leader.emotion_management_ability = 'None'
        leader.intervention_threshold = None  # changed from -0.3 to None because this one doesn't intervene at all (used for base comparison)

    return leader
