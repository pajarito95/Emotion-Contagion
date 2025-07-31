import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


populationSize = 11  # Bosse et al. (2015) used 10 agents, so we use 11 to pick out a leader and have 10 agents interacting with each other
style = "High"  # designate "High" or "Low" to indicate the leader's emotion management ability


def save_results(avg_emotion_runs, folderpath, images=None):
    '''
    Save simulation results and any associated plots (save each scenario).

    Parameters:
    - avg_emotion_runs: list of lists of average emotion history (one list per run)
    - folderpath: path to the parent save directory
    - images: optional list of (matplotlib.figure.Figure, filename) tuples
    '''
    fullpath = os.path.join(folderpath, "Final Results")
    os.makedirs(fullpath, exist_ok=True)

    # Save as DataFrame with time steps as index, runs as columns
    df = pd.DataFrame(avg_emotion_runs).T
    df.columns = [f"Run_{i+1}" for i in range(len(avg_emotion_runs))]
    df.index.name = "Time"
    df.to_csv(os.path.join(fullpath, f"Avg_Emotion_Evolution_Over_30_Simulations_{style}.csv"))

    if images:
        for fig, filename in images:
            figpath = os.path.join(fullpath, filename)
            fig.savefig(figpath)
            fig.clf()

    print(f"Simulation results saved to {fullpath}")



def initialize(seed=None):
    '''
    Initialization function of simulation environment -- setup the variables.
    (D) = dynamic parameter (continuously updated) and (S) = static parameter (fixed)
    The intervals used have a uniform/gaussian distribution.
    Agent parameters:
        Emotion (D): How an agent feels towards the change. Defined within the interval [-1, 1] following a beta distribution to skew towards the negative side.
                     The purpose behind the interval is because we want to see how a leader's emotional aperature dictates the evolution of the emotion contagion which gives more meaningful results when emotions initially tend more negative (encourages the random number picker towards negative).
        Expressiveness (S): The degree to which an agent would display their emotion. [0,1]
        alpha (S): Susceptibility to being influenced. This can also be thought of as how resistant an agent may be (stubborness, strong-headedness, etc.)
        Amplification (D): How much an agent's emotion is amplified when interacting with another agent. This is a combination of other parameters
        Bias (D): How much an agent's emotion is biased towards similar emotions. This is a combination of other parameters
        Transmission (D): How much an agent's emotion is received by another agent during interaction. This is a combination of other parameters
    Leader parameters:
        Emotion (S): Value = 1. We assume the leader is completely on board (feels fully positive) about the change.
        Charisma (S): Leader's ability to influence the team. This is a value between 0 and 0.5, arbitrarily chosen by us
        emotionManagementAbility (S): Leader's ability to manage the sentiments amongst the team [H,L]. H = high ability, L = low ability
        interventionThreshold (S): what the average sentiment amongst the team needs to be in order for the leader to intervene 
    Other:
        intimacyMatrix (S): Represents agent-to-agent relationships (how close they are to each other) [0,1]. Explicitly indicates probability of an interaction between a pair of agents (one-way intitiation). 
                            Leader is currently excluded from the assigment of intimacy between it and it's team members.
                            Intimacies are asymmetrical amongst agent pairs (i.e. how close agentA feels to agentB and how close agentB feels to agentA need not be the same).
    '''
    global agents, leader, intimacyMatrix, emotionHistory, avgEmotionHistory  # create global variables so that they are readily accessible outside of this function (on the basis that this function is called before the variables within are needed to be used elsewhere)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    emotionHistory = []  # to store the emotions of each agent at each time step (for sentiment evolution graph and sentiment segregation visual)
    avgEmotionHistory = []  # to store the average emotion of the agents at each time step (for sentiment evolution graph)

    agents = []  # initialize list of agents -- simple and dynamic data structure, e.g. lists can change in size (meaning agents added or removed during simulation -- though we are not doing this); order is preservered, thus ensuring agents are consistently/simply iterated over
    for _ in range(populationSize):  # for each agent in the population
        newAgent = {'Emotion': -1 + 2 * np.random.beta(2, 5), 
                    'alpha': random.uniform(0, 0.2),
                    'Expressiveness': random.uniform(0,1),
                    'Amplification': random.uniform(0,1),
                    'amplificationBias': random.uniform(0,1)}  # dictionary for parameter readability
        agents.append(newAgent)  # add agent to list
    
    leader = random.choice(agents)  # randomly select an agent from the list to designate as leader 
    agents.remove(leader)  # remove it from the agents list as they will behave differently and their parameters are going to be changed below


    ### Update leader parameters
    del leader['alpha']  # remove this and below as they are not needed for the leader
    del leader['Expressiveness']
    del leader['Amplification']
    del leader['amplificationBias']

    leader['Emotion'] = 1  # fix leader emotion
    
    # add parameters
    emotionManagementAbility = style
    if emotionManagementAbility == 'High':
        leader.update({'emotionManagementAbility': 'High'})
        leader.update({'interventionThreshold': -0.2})  # we're only playing around with this aspect of leader aperture, so we only vary this parameter
        leader.update({'Charisma': 0.25})
    else:
        leader.update({'emotionManagementAbility': 'Low'})
        leader.update({'interventionThreshold': -0.6})
        leader.update({'Charisma': 0.25})

    ### Leader-agent weights -- future consideration
    leaderAgentWeights = np.random(random.uniform(0,1), (populationSize, 1))

    for idx, agent in enumerate(agents):
        agent['index'] = idx  # add an index to each agent for easy reference later on (e.g. in the social

    ### Intimacy weights (agent-wise)
    intimacyMatrix = np.random.uniform(0,1, (populationSize, populationSize))
    np.fill_diagonal(intimacyMatrix, 0)  # along the diagonal are the self-to-self pairs (agentA, agentA), so we may assign a weight of 0
    intimacyMatrix = intimacyMatrix/intimacyMatrix.sum(axis=1, keepdims=True)  # normalise the matrix so we could treat these directly as probabilities (for what idk, but it is a thought)



def emotional_valence_update(agentA, agentB, agentA_index, agentB_index):  # are the matrix and dictionary indices aligning?
    '''
    For updating the emotional valence of each agent when they interact with the other agents.
    For contagionStrength, one may also think of it as influence strength
    '''
    AB_intimacy = intimacyMatrix[agentA_index,agentB_index]
    BA_intimacy = intimacyMatrix[agentB_index,agentA_index]

    agentA['Emotion'] += agentA['alpha']*AB_intimacy*(agentB['Emotion'] - agentA['Emotion'])
    agentA['Emotion'] = np.clip(agentA['Emotion'], -1, 1)  # clip the emotion to be within [-1, 1]      
    
    agentB['Emotion'] += agentB['alpha']*BA_intimacy*(agentA['Emotion'] - agentB['Emotion'])
    agentB['Emotion'] = np.clip(agentB['Emotion'], -1, 1)  # clip the emotion to be within [-1, 1]



def avgEmotion(agents):
    '''
    Calculate average emotion valence amongst the team.
    '''
    return (sum(agent['Emotion'] for agent in agents))/max(1, len(agents))  # the max(1,...) is there in case agents is 0 or empty for whatever reason



def agent_interaction():
    '''
    Define inter-agent interactions. 
    Since their intimacy is defined using a probabilty, we say if that is a above a randomly generated number [0,1], that is an interaction.
    '''
    global agents, emotionHistory, avgEmotionHistory

    buddies = []
    for i, agentA in enumerate(agents):  # for each index i and the agent, agentA, at that index in agents list (NOTE: using enumerate() allows us to easily get the indices in the agents list which we later use to access the agent objects)
        for j, agentB in enumerate(agents):  # same as above but we use a different letter to help differentiate from above because
            if i != j:  # we of course do not want to select the same agent haha
                interaction_prob = max(intimacyMatrix[int(i),int(j)], intimacyMatrix[int(j),int(i)])  #  get the intimacy of the agent with the stronger intimacy
                if random.random() < interaction_prob:  # if that intimacy is greater than some random number (could also use <= instead of <) -- this increases the stochasticness of interactions
                    buddies.append((i,j))  # we shall define this as an interaction

    for i,j in buddies:  # for each pair i,j in our buddies list
        agentA, agentB = agents[i], agents[j]  # get the corresponding agent objects
        emotional_valence_update(agentA, agentB, i, j)  # update the emotional valence of both



def leader_intervention():
    '''
    Here we define when a leader intervenes and what happens when they do (update the Emotion of agents). 
    We use a leaderImpact value to specify the degree of their intervention. This is currently some arbitrary value greater than 0.
    We do not have to specify what the intervention is, just the degree that it impacts the team and here we may get creative.
    '''
    global time, agents, leader, avgEmotionValence

    print(f"Leader Ability: {leader['emotionManagementAbility']}")
    # leader intervenes when the average emotion is at that leaders minimum intervention threshold level
    if avgEmotionValence <= leader['interventionThreshold']:  
        for agent in agents:
            agent['Emotion'] += agent['alpha']*leader['Charisma']*(leader['Emotion'] - agent['Emotion'])  # multiplying by an agents susceptibility makes the intervention impact vary per agent
            agent['Emotion'] = np.clip(agent['Emotion'], -1, 1)  # clip the emotion to be within [-1, 1]


def sentiment_evolution_graph(intervention_timesteps):
    '''
    A graph to show the evolution of all emotions throughout the time steps (essentially a time series!). That is, the emotions from start to end.
    NOTE: mark when leader intervenes (x-axis)
    '''
    emotion_array = np.array(emotionHistory)  # shape: (timesteps, num_agents)
    avg_array = np.array(avgEmotionHistory)
    number_of_agents = emotion_array.shape[1]
    flat_interventions = sorted(set(intervention_timesteps))  # flatten the list of lists and sort it

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(number_of_agents):
        ax.plot(emotion_array[:, i], alpha=0.6, color='gray')
        # cmap = plt.get_cmap('tab10')  # Use a colormap for distinguishable colors
        # ax.plot(emotion_array[:, i], alpha=0.6, color=cmap(i % cmap.N))  # Assign a unique color to each agent

    for t in flat_interventions:
        ax.axvline(x=t, color='tab:blue', linestyle='--', alpha=0.5, label='Leader Intervention' if t == flat_interventions[0] else "") # mark the intervention times with a vertical line

    ax.plot(avg_array, color='red', linewidth=2, label='Average Emotion')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Emotion Value')
    ax.set_title('Sentiment Dynamics Over Time')
    ax.grid(True)
    ax.set_ylim(-1, 1)
    ax.legend()
    plt.show(block=False)
    fig.tight_layout()

    return fig



def social_network_graph():
    '''
    To create a social network graph representing the agent interactions throughout the simulation.
    Idea 1 (undirected): edges between agents represent how frequently that pair interacted (intensity=frequency) and hopefully which agents were the most influential/popular
    Idea 2 (directed): two edges maximum allowed between agents (one in each direction) representing the exchange of information from one agent to the other
    '''

    # 1) interaction frequency graph
    G = nx.Graph()  # create a new graph object
    G.add_nodes_from([agent['index'] for agent in agents])  # adding nodes where each represents an agent via it's index in the agents list
    edge_list = 
    


    # 2) information exchange graph
    DG = nx.DiGraph()
    DG.add_nodes_from(G)  # want the same nodes, just different edges



def run_simulation(seed=0, run_id=0, save_folder=None):
    '''
    Calling all functions as necessary and storing the results and associated parameters
    '''
    global time, avgEmotionValence
    
    initialize(seed=seed)  # set up environment
    print(f"Leader ability: {leader['emotionManagementAbility']}")
    
    intervention_timesteps = []  # to store the time steps at which the leader intervenes
    emotionHistory.append([agent['Emotion'] for agent in agents])  # add each agent's emotion to this list for tracking over time -- doing it here will get us their initial states (before agent interaction or leader intervention)

    max_iterations = 500
    time = 0
    while time < max_iterations+1:  # while both conditions of convergence not reached and the max iterations is also not reached, the simulation continues (when at least one of the conditions is met, then it will stop)
        agent_interaction()
        avgEmotionValence = avgEmotion(agents)
        avgEmotionHistory.append(avgEmotionValence)

        #if the avg emotion valence falls within the leaders intervention threshold
        if avgEmotionValence <= leader['interventionThreshold']:
            leader_intervention()
            print("Leader intervening")
            intervention_timesteps.append(time)

        emotionHistory.append([agent['Emotion'] for agent in agents])  # doing it here will get us their emotions as the simulation continues
        print(f"Time {time} - Avg Emotion: {avgEmotionValence}")
        time += 1

    fig = sentiment_evolution_graph(intervention_timesteps)  # create time graph at end of simulation to show the evolution
    images = [(fig, f"Sentiment_Evolution_Run{run_id+1}_Seed{seed}.png")]  # list of images to save

    leader_data = {
        'emotionManagementAbility': leader.get('emotionManagementAbility'),
        'interventionThreshold': leader.get('interventionThreshold'),
        'finalAvgEmotion': avgEmotionHistory[-1] if avgEmotionHistory else None,
        'finalAgentEmotions': [agent['Emotion'] for agent in agents],
    }

    results = {
        'leader': leader_data,
        'emotion_history': emotionHistory,
        'avg_emotion_history': avgEmotionHistory,
        'intervention_timesteps': intervention_timesteps,
        'final_avg_emotion': avgEmotionValence,
    }

    if save_folder:
        run_folder = os.path.join(save_folder, f"Run_{run_id + 1:02d}")
        os.makedirs(run_folder, exist_ok=True)
        save_results([avgEmotionHistory], run_folder, images)
    
    return results


basefolder = r"\Masters\523\Simulation\same2"  # in the quotations, put the name of the main folder in which you want the results to be saved to  
folderpath = os.path.join(basefolder, f"Leader_{style}")  # create a folder for the leader's emotion management ability
def run_multiple_simulations(runs=30, save_folder=folderpath):
    '''
    Run multiple simulations and save the results
    '''
    all_avg_emotion_list = []
    all_interventions_list = []
    for run in range(runs):
        print(f"Running simulation {run+1}/{runs}")
        global agents, leader, emotionHistory, avgEmotionHistory  # reset the global variables for each run
        agents = []
        leader = {'Emotion': 1, 'emotionManagementAbility': style, 'interventionThreshold': -0.2}
        emotionHistory = []
        avgEmotionHistory = []
        results = run_simulation(seed=run, run_id=run, save_folder=folderpath)  # run the simulation with a different seed for each run
        all_avg_emotion_list.append(results['avg_emotion_history'])
        all_interventions_list.append(results['intervention_timesteps'])
    
    save_results(
        avg_emotion_runs=all_avg_emotion_list,  # Pass the list of avg_emotion_history for all runs
        folderpath=save_folder
    )

    return all_avg_emotion_list, all_interventions_list



os.makedirs(folderpath, exist_ok=True)
runs = 30
all_avg_emotion_list, all_interventions_list = run_multiple_simulations(runs=runs)

# save intervention data
interventionpath = os.path.join(folderpath, f"Intervention")
os.makedirs(folderpath, exist_ok=True)
df_interventions = pd.DataFrame({
    "Run": list(range(1, runs + 1)),
    "Intervention_Timesteps": [','.join(map(str, run)) for run in all_interventions_list],
    "Num_Interventions": [len(run) for run in all_interventions_list]
})

binary_matrix = np.zeros((runs, 501), dtype=int)  # 501 timesteps for full duration: Time 0 to Time 500
for i, run in enumerate(all_interventions_list):
    for t in run:
        binary_matrix[i, t] = 1

df_bool = pd.DataFrame(binary_matrix).T  # shape: (501, runs)
df_bool.to_csv(os.path.join(folderpath, f"Intervention_BooleanMatrix_{style}.csv"), index=False)