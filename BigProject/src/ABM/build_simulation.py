from .agents import make_agents, select_leader, configure_leader
from .simulation_state import SimulationState
from .network import create_intimacy_matrix

def initialize_simulation(rng, population_size, structure, intra_strength, inter_strength, leader_style):
    agents = make_agents(rng, population_size)
    leader, agents = select_leader(rng, agents)
    configure_leader(leader, leader_style)

    intimacy_matrix, assignments = create_intimacy_matrix(
        rng,
        population_size,
        structure,
        intra_strength,
        inter_strength,
    )

    # Leader's intimacy is the average of the team's intimacy
    leader_intimacy = intimacy_matrix.mean(axis=0)

    return SimulationState(
        agents=agents,
        leader=leader,
        intimacy_matrix=intimacy_matrix,
        leader_intimacy=leader_intimacy,
        assignments=assignments
    )