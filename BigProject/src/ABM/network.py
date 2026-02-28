import numpy as np

def create_intimacy_matrix(
    rng,
    population,
    structure,
    intra_strength,  # within (arbitrary placeholder values)
    inter_strength,  # between
    core_to_core=0.65,      # core-core (those within the core influence); also idk how i feel with these weights; how should we even do them?
    core_to_periph=0.5,     # core -> periphery (core to periphery influence)
    periph_to_core=0.2,     # periphery -> core (periphery to core ifluence)
    periph_to_periph=0.2,    # periphery-periphery (those within the periphery influence)
    core_proportion=0.25,    # what percentage of nodes do we want in the core
    size=None,
    min_weight=0.01,
):
    """
    Create an asymmetric, row-normalized intimacy matrix for two communities, random, or core-periphery structure.
    For random, intra_strength and inter_strength should be the same and size is ignored. The strengths function as a cap/upper bound for the (random) intimacy values
    
    TODO: community is currently set up for two communities, but can be made more dynamic by changing the block structure and assignment logic.
    """
    assert structure in ["community", "random", "core-periphery"], "Structure must be 'community', 'random', or 'core-periphery'."
    assert not (structure == "random" and (intra_strength != inter_strength)), "For random structure, intra_strength and inter_strength must be the same."
    assert population % 2 == 0, "Population (team size) must be even."
    assert 0 < min_weight < 1, "min_weight must be between 0 and 1."
    assert (0 <= intra_strength <= 1 and 0 <= inter_strength <= 1) or (0 <= core_to_core <= 1 and 0 <= core_to_periph <= 1 and 0 <= periph_to_core <= 1 and 0 <= periph_to_periph <= 1), "Strength values must be given and must between 0 and 1."
    

    if structure == "community" or structure == "random":
        if size is None:
            n1 = population // 2
            n2 = population - n1
        else:
            n1, n2 = size
            assert n1 + n2 == population, "size must sum to population."
        
        # Randomly assign communities: 0 and 1
        assignments = np.ones(population, dtype=int)
        community0_indices = rng.choice(population, size=n1, replace=False)
        assignments[community0_indices] = 0  # 0 = community 0, 1 = community 1

        # Block upper bounds
        block_bounds = np.array([
            [intra_strength, inter_strength],
            [inter_strength, intra_strength]
        ])

        W = np.zeros((population, population), dtype=float)
        for i in range(population):
            for j in range(population):
                bound = block_bounds[assignments[i], assignments[j]]
                assert bound >= min_weight, "min_weight exceeds upper bound."
                W[i, j] = rng.uniform(min_weight, bound)

        # Normalize so each row sums to 1
        intimacyMatrix = W / W.sum(axis=1, keepdims=True)

        return intimacyMatrix, assignments
    

    elif structure == "core-periphery":
        assert 0 <= core_proportion <= 1
        core_size = max(1, round(population * core_proportion))  # at least 1 core node
        #periph_size = population - core_size  # comment this out for random assignment
        
        # Assign first `core_size` as core (1), rest as periphery (0)
        #assignments = np.array([1]*core_size + [0]*periph_size)  # comment this out for random assignment
        
        # Or randomly assign core/periphery
        core_indices = rng.choice(population, size=core_size, replace=False)
        assignments = np.zeros(population, dtype=int)
        assignments[core_indices] = 1  # 1 = core, 0 = periphery

        # block bounds: [from_type, to_type]
        # 1 = core, 0 = periphery
        block_bounds = np.array([
            [periph_to_periph, periph_to_core],
            [core_to_periph, core_to_core]
        ])
        
        W = np.zeros((population, population))
        for i in range(population):
            for j in range(population):
                bound = block_bounds[assignments[i], assignments[j]]
                assert bound >= min_weight, "min_weight exceeds an upper bound; lower it."
                W[i, j] = rng.uniform(min_weight, bound)
        
        # row normalize
        intimacyMatrix = W/W.sum(axis=1, keepdims=True)
        
        return intimacyMatrix, assignments
    