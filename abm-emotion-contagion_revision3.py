import pycxsimulator
from pylab import *
import random
import numpy as np

# number of agents (people in this case)
populationSize = 11
# NOTE: if it is desired to have 10 agents interacting with each other, we could increase the above by +1 because currently we start with 10 and pick out leader, thus reducing the team to 9 agents who interact amongst each other

# This parameter/variable is fixed for now, though we may consider if we can develop some be more dynamic method for determining it later
posConvergenceThreshold = 0.6  # Convergence limit. That is, at a certain point we may declare that a desired level of emotion contagion convergence has been reached


# Considerations:
# employementTime as parameter where lower time = more likely to stay positive ~ could be related to expressiveness
# resignation/acceptance
# weights between leader and agents so agents are affected in a dynamic way from leader intervention


##############
folderpath = r""  # in the quotations, put the name of the folder in which you want the results to be saved to  
def save_results(results, folderpath, images=None):
    '''
    To save the results
    TODO: it's incomplete
    '''
    import os
    import pandas as pd

    subfolders = [name for name in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath)) and name.startswith("Results")]

    if subfolders:
        runs = []
        for name in subfolders:
            try:
                run = int(name.split("Results")[-1].strip())
                runs.append(run)
            except ValueError:
                continue
        latest_run = max(runs) + 1
    else:
        latest_run = 0
    
    new_subfolder_name = f"Results {new_number}"
    fullpath = os.path.join(folderpath, new_subfolder_name)
    os.makedirs(fullpath, exist_ok=True)

    # we oughta look into how Results is being formatted
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(fullpath, "Results.csv"), index=False)

    if images:
        for fig, filename in images:
            figpath = os.path.join(fullpath, filename)
            fig.savefig(figpath)
            fig.clf()

    print(f"Simulation results saved to {fullpath}")


"""
Case 1: 
    Emotion: uniform(-0.5, 1)
    Leader alpha: 1
    emotionManagementAbility: H
    interventionThreshold: -0.2

Case 2: 
    Emotion: uniform(-0.5, 1)
    Leader alpha: 1
    emotionManagementAbility: L
    interventionThreshold: -0.8

Case 3: 
    Emotion: uniform(-1, 0)
    Leader alpha: 1
    emotionManagementAbility: H
    interventionThreshold: -0.2

Case 4: 
    Emotion: uniform(-1, 0)
    Leader alpha: 1
    emotionManagementAbility: L
    interventionThreshold: -0.8
"""


def initialize():
    '''
    Initialization function of simulation environment -- setup the variables.
    (D) = dynamic parameter (continuously updated) and (S) = static parameter (fixed)
    The intervals used have a uniform/gaussian distribution.
    Agent parameters:
        Emotion (D): How an agent feels towards the change. Defined within the interval [-1, n] where n is some arbitrary positive value less than 1. Initially randomly assigned and then dynamically updated during simulation. 
                     The purpose behind the interval is because we want to see how a leader's aperature dictates the evolution of the emotion contagion which gives more meaningful results when emotions initially tend more negative. This will also encourage the random number picker towards negative.
                     NOTE: Is a cap needed to prevent the Emotion parameter from going outside [-1, 1] during the agent_interaction() and leader_interaction() functions?
        Expressiveness (S): How expressive an agent is. This is a static parameter that is randomly assigned to each agent.
        alpha (S): Susceptibility to being influenced. This can also be thought of as how resistant an agent may be (stubborness, strong-headedness, etc.)
        gamma (S): Susceptibility to group influence. Same as above but this is how susceptible an agent is to the group as a whole. This is not used in the current version of the code, but we may want to consider it later.
                   Assigned 0 initially, but soon updated with a calculated value using alpha and intimacyMatrix.
    Leader parameters:
        Emotion (S): Value = 1. We assume the leader is completely on board (feels fully positive) about the change.
        alpha (S): Value = 0. We assume the leader is not influenced by their team members.
        emotionManagementAbility (S): Leader's ability to manage the sentiments amongst the team [H,L]. H = high ability, L = low ability
        interventionThreshold (S): what the average sentiment amongst the team needs to be in order for the leader to intervene 
    Other:
        intimacyMatrix (S): Represents agent-to-agent relationships (how close they are to each other) [0,1]. Explicitly indicates probability of an interaction between a pair of agents (one-way intitiation). 
                            Leader is currently excluded from the assigment of intimacy between it and it's team members.
                            Intimacies are asymmetrical amongst agent pairs (i.e. how close agent_r feels to agent_c and how close agent_c feels to agent_r need not be the same).
    '''
    global agents, leader, intimacyMatrix, emotionHistory, avgEmotionHistory  # create global variables so that they are readily accessible outside of this function (on the basis that this function is called before the variables within are needed to be used elsewhere)

    emotionHistory = []  # to store the emotions of each agent at each time step (for sentiment evolution graph and sentiment segregation visual)
    avgEmotionHistory = []  # to store the average emotion of the agents at each time step (for sentiment evolution graph)

    agents = []  # initialize list of agents -- simple and dynamic data structure, e.g. lists can change in size (meaning agents added or removed during simulation -- though we are not doing this); order is preservered, thus ensuring agents are consistently/simply iterated over
    for _ in range(populationSize):  # for each agent in the population
        newAgent = {'Emotion': uniform(-1,0), 'Expressiveness': uniform(0,1),'alpha': uniform(0,1), 'gamma': 0}  # dictionary for parameter readability
                                        # ^ Here we have Emotion take on -1 to some arbitrary positive value less than 1 since we want to see how a leader influences agents when there is some negativity abound 
        agents.append(newAgent)  # add agent to list


    leader = random.choice(agents)  # randomly select an agent from the list to designate as leader 
    agents.remove(leader)  # remove it from the agents list as they will behave differently and their parameters are going to be changed below

    ### Update leader parameters
    leader['Emotion'] = 1  # set their emotion = 1
    del leader['alpha']  # delete alpha and gamme since we not using them for the leader -- we actually don't need to do this since they're not used at all, but why not remove them for cleanliness
    del leader['gamma']


    #leader.update({'emotionManagementAbility': 'H'})
    #leader.update({'interventionThreshold': -0.2})
    
    leader.update({'emotionManagementAbility': 'L'})
    leader.update({'interventionThreshold': -0.8})


    # leader.update({'emotionManagementAbility': random.choice(['H','L'])})  # add emotionManagementAbility parameter 
    # add arbitrary intervention thresholds
    # if leader['emotionManagementAbility'] == 'H':
    #     leader.update({'interventionThreshold': -0.05})
    # else:
    #     leader.update({'interventionThreshold': -0.4})



    ### Intimacy weights (agent-wise)
    intimacyMatrix = np.random.uniform(0,1, (populationSize, populationSize))
    np.fill_diagonal(intimacyMatrix, 0)  # along the diagonal are the self-to-self pairs (agent_r, agent_r), so we may assign a weight of 0
    intimacyMatrix = intimacyMatrix/intimacyMatrix.sum(axis=1, keepdims=True)  # normalise the matrix so now we may treat these directly as probabilities!


    for agent_index, agent in enumerate(agents):  # for each agent in the agents list with its index
        agent['gamma'] = agent['alpha'] * np.sum(intimacyMatrix[agent_index])  # update the gamma parameter for each agent based on their intimacy with the other agents


    ### Leader-agent weights -- if desired later
    #leaderAgentWeights = np.random(uniform(0,1), (populationSize, 1))


def emotional_valence_update(agent_r, agent_c, agent_r_index, agent_c_index):
    '''
    For updating the emotional valence of each agent when they interact with the other agents.
    NOTE: Formula needs revision
    '''  
    # the weight is intimacy from agent_r to all other agents
    q_r_sum = sum(intimacyMatrix[agent_r_index, agent_s_index]*agent_s['Emotion'] for agent_s_index, agent_s in enumerate(agents) if agent_s != agent_r) # sum of the emotions of all agents (excluding agent_r) weighted by tagent_r's intimacy to that agent
    q_r_star = q_r_sum/(len(agents)-1)  # average emotion of the other agents (excluding agent_r)
    agent_r['Emotion'] += agent_r['alpha']*intimacyMatrix[agent_r_index,agent_c_index]*(agent_c['Emotion'] - agent_r['Emotion']) #+ agent_r['gamma']*(q_r_star - agent_r['Emotion'])
    agent_r['Emotion'] = np.clip(agent_r['Emotion'], -1, 1)  # clip the emotion to be within [-1, 1]

    q_c_sum = sum(intimacyMatrix[agent_c_index, agent_s_index]*agent_s['Emotion'] for agent_s_index, agent_s in enumerate(agents) if agent_s != agent_c)  # sum of the emotions of all agents (excluding agent_c) weighted by agent_c's intimacy to that agent
    q_c_star = q_c_sum/(len(agents)-1)  # average emotion of the other agents (excluding agent_c)
    agent_c['Emotion'] += agent_c['alpha']*intimacyMatrix[agent_c_index,agent_r_index]*(agent_r['Emotion'] - agent_c['Emotion']) #+ agent_c['gamma']*(q_c_star - agent_c['Emotion'])
    agent_c['Emotion'] = np.clip(agent_c['Emotion'], -1, 1)  # clip the emotion to be within [-1, 1]

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
    for i, agent1 in enumerate(agents):  # for each index i and the agent, agent1, at that index in agents list (NOTE: using enumerate() allows us to easily get the indices in the agents list which we later use to access the agent objects)
        for j, agent2 in enumerate(agents):  # same as above but we use a different letter to help differentiate from above because
            if i != j:  # we of course do not want to select the same agent haha
                interaction_prob = max(intimacyMatrix[int(i),int(j)], intimacyMatrix[int(j),int(i)])  #  get the intimacy of the agent with the stronger intimacy
                if random.random() < interaction_prob:  # if that intimacy is greater than some random number (could also use <= instead of <) -- this increases the stochasticness of interactions
                    buddies.append((i,j))  # we shall define this as an interaction

    for i,j in buddies:  # for each pair i,j in our buddies list
        agent1, agent2 = agents[i], agents[j]  # get the corresponding agent objects
        emotional_valence_update(agent1, agent2, i, j)  # update the emotional valence of both
        

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
            agent['Emotion'] += agent['alpha']*(leader['Emotion'] - agent['Emotion'])  # multiplying by an agents susceptibility makes the intervention impact vary per agent
            agent['Emotion'] = np.clip(agent['Emotion'], -1, 1)  # clip the emotion to be within [-1, 1]


def sentiment_segregation_visual():
    '''
    A pretty picture to show the evolution of agents' emotions over time. This would be the emotions at a time t.
    NOTE: We should set the pictures to be saved.
    Blue = negative (<0)
    Orange = positive (>0)
    Gray = neutral (=0) (unlikely, but technically possible)
    '''
    import matplotlib.pyplot as plt

    colors = []
    for agent in agents:
        if agent['Emotion'] < 0:
            colors.append('blue')
        elif agent['Emotion'] > 0:
            colors.append('orange')
        else:
            colors.append('gray')

    # Assign arbitrary coordinates for plotting purposes
    ### Ask Shayan for help in plotting this. The plots in his presentation looked so pretty
    x = [random.uniform(0,100) for agent in range(len(agents))]
    y = [random.uniform(0,100) for agent in range(len(agents))]

    plt.figure(figsize=(10, 2))
    plt.scatter(x, y, c=colors, s=100)
    plt.title(f'Sentiment Segregation at timestep {time}')  # i think we need to verify when to call this function because i think it was that we had two graphs for timestep = 0?
    plt.axis('off')
    plt.savefig(f'sentiment_segregation_timestep_{time}.png', bbox_inches='tight')
    # plt.show(block=False)  # so simulation continues while showing plot
    # plt.pause(3)  # let the graph stay up for us to see for 3 seconds
    plt.close()




def sentiment_evolution_graph():
    '''
    A graph to show the evolution of all emotions throughout the time steps (essentially a time series!). That is, the emotions from start to end.
    NOTE: mark when leader intervenes (x-axis)
    '''
    emotion_array = np.array(emotionHistory)  # shape: (timesteps, num_agents)
    avg_array = np.array(avgEmotionHistory)
    number_of_agents = emotion_array.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(number_of_agents):
        plt.plot(emotion_array[:, i], alpha=0.6, color='gray')  # we could pick a different set of colors if we want something other than gray/red. Perhaps a colorblind friendly combination

    plt.plot(avg_array, color='red', linewidth=2, label='Average Emotion')
    plt.xlabel('Time Step')
    plt.ylabel('Emotion Value')
    plt.title('Sentiment Dynamics Over Time')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)


def social_network_graph():
    '''
    To create a social network graph representing the agent interactions throughout the simulation.
    Idea 1 (undirected): edges between agents represent how frequently that pair interacted (intensity=frequency) and hopefully which agents were the most influential/popular
    Idea 2 (directed): two edges maximum allowed between agents (one in each direction) representing the exchange of information from one agent to the other. Or perhaps this could/should be undirected?
    '''
    import matplotlib.pyplot as plt
    import networkx as nx

    G_un = nx.Graph()  # create a new graph object
    G_di = nx.DiGraph()



def run_simulation():
    '''
    Calling all functions as necessary and storing the results and associated parameters
    '''
    global time, avgEmotionValence
    time = 0
    initialize()  # set up environment
    print(f"Leader ability: {leader['emotionManagementAbility']}")
    
    emotionHistory.append([agent['Emotion'] for agent in agents])  # add each agent's emotion to this list for tracking over time -- doing it here will get us their initial states (before agent interaction or leader intervention)

    sentiment_segregation_visual()  # create the emotion segregation before leader intervention
    
    max_iterations = 250
    convergence = False
    # while both conditions of convergence not reached and the max iterations is also not reached, the simulation continues (when at least one of the conditions is met, then it will stop)
    while not convergence and time < max_iterations+1:
        agent_interaction()
        avgEmotionValence = avgEmotion(agents)
        avgEmotionHistory.append(avgEmotionValence)
        sentiment_segregation_visual()

        # if the avg emotion valence falls within the leaders intervention threshold
        if avgEmotionValence <= leader['interventionThreshold']:
            leader_intervention()
            print("Leader intervening")

        emotionHistory.append([agent['Emotion'] for agent in agents])  # doing it here will get us their emotions as the simulation continues

        # if the average emotion is sufficient as per our pre-defined acceptable level, we may say this a convergence has been reached
        if avgEmotionValence >= posConvergenceThreshold:
            convergence = True
        print(f"Time {time} - Avg Emotion: {avgEmotionValence}, Convergence: {convergence}")

        time += 1

    sentiment_evolution_graph()  # create time graph at end of simulation to show the evolution

    leader_data = {
        'emotionManagementAbility': leader.get('emotionManagementAbility'),
        'interventionThreshold': leader.get('interventionThreshold'),
        'finalAvgEmotion': avgEmotionHistory[-1] if avgEmotionHistory else None,
        'finalAgentEmotions': [agent['Emotion'] for agent in agents],
    }

    # Package up the results  -- we should add in code to save the results or explicitly output them (and we also need to have the graphs saved as well) so we have access to them
    results = {
        'leader': leader_data,
        'emotion_history': emotionHistory,
        'avg_emotion_history': avgEmotionHistory
    }

    return results

run_simulation()









