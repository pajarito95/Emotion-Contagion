import pycxsimulator
from pylab import *
from random import randint, uniform, choice
import numpy as np

# grid parameters, number of agents (people in this case), collision
width = 50  
height = 50
populationSize = 10  # per Dionne et al., 2010
cr = 0.02  # radius for detecting a "collision" between agents (how close agents have to be for it to be considered an interaction)
crsq = cr**2

# These parameters/variables are fixed for now, though we may consider having some be more dynamic later
sentimentStrength = 0.02  # how strong a sentiment is
affectCoefficient = 0.8  # how much it affects others. Could consider letting it vary later. How will this connect with intimacy?
contagionProb = 0.95  # how to define this? Could be the probability of an agent's sentiment being affected? Could be the probability of an agent's sentiment affecting another? This one would be multiplied by the intimacy
emotionManagementAbility = uniform(0,1)  # leader agent's emotion management ability is randomly chosen within the interval 
posConvergenceThreshold = 0.8  # convergence limit. That is, at a certain point (one for negative and one for positive) we may declare that an emotion contagion convergence has been reached
negConvergenceThreshold = 0.2  # or perhaps we may include divergences?

# Considerations:
# employementTime as parameter where lower time = more likely to stay positive
# resignation/acceptance
# da Silva Costa (2022) offered a few concern-reduction strategies such as ambassadors and information sessions

def initialize():
    '''
    Initialization function of simulation environment. 
    Setup the variables.
    An agent also had a fixed parameter, alpha, indicating susceptibility to being influenced, along with a default state of expressiveness (the overall intimacy score, to be calculated later)
    Where (D) = dynamic parameter (continuously updated) and (S) = static parameter (fixed)
    Agent parameters:
        x, y: coordinates
        Emotion (D): How an agent feels towards the change. Defined as fully negative to fully positive [-1,1]. Initially randomly assigned and get's dynamically updated throughout simulation.
        alpha (S): Susceptibility to change, i.e. how likely they are to be in agreement with it. This can also be thought of as how resistant an agent may be -- WAIT: is [0,1] okay?
        Expressiveness (S): How often an agent would represent their emotion. It is an overall intimate value calculated by summing all their connections (intimacies) to other agents. 
                        If using option 1, calculations must be performed using intimacies from agent_r to agent_c.
    Leader parameters:
        Emotion = 1: We assume the leader is completely on board (feels fully positive) about the change.
        Intimacy: DNE. Leader is excluded from the assigment of intimacy between it and it's team members.
        alpha = 0: We assume the leader is not influenced by their team members.
        emotionManagementAbility (S): Leader's ability to manage the sentiments amongst the team [0,1]. We assume 0 management means the leader does not attempt to intervene, otherwise, any intervention is positive
        interventionThreshold (S): what the average sentiment amongst the team needs to be in order for the leader to intervene 
    Other:
        Intimacy (S): Represents agent-to-agent relationships (how close they are to each other) [0,1]. Indicates liklihood of an interaction between a pair of agents and consequently how likely this is to influence their emotion (along with the alpha parameter).
            Option 1: Increase complexity with asymmetrical intimacies amongst agent pairs (i.e. how close agent_r feels to agent_c and how close agent_c feels to agent_r need not be the same)
            Option 2: Simplify complexity with symmetric intimacies amongst agent pairs (i.e. how close agent_r and agent_c are to eachother is equal in both directions)
    '''
    global time, agents, leader, intimacyMatrix, envir, nextenvir  # create global variables; nextenvir is a temporary array for computing the next state in order to update the main environment, envir

    time = 0  # time step starts at 0
    
    agents = []  # initialize list of agents -- simple and dynamic data structure, e.g. lists can change in size (meaning agents added or removed during simulation -- though we are not doing this); order is preservered, thus ensuring agents are consistently/simply iterated over
    for agent in range(populationSize):  # for each agent in the population
        # randint(width) and randint(height) respectively create random x and y coordinates (within our predefined width/height range)
        #newAgent = [randint(width), randint(height), random.uniform(-1,1), random.uniform(0,1), 0]  # list for simplicity/directness
        newAgent = {'x': randint(width), 'y': randint(height), 'Emotion': uniform(-1,1), 'alpha': uniform(0,1), 'Expressiveness': 0}  # dictionary for parameter readability
        agents.append(newAgent)  # add agent to list

    leader = random.choice(agents)  # randomly select an agent from the list to designate as leader 
    agents.remove(leader)  # exclude it from the rest as their sentiment (positive) is fixed


    ### update leader parameters
    leader['Emotion'] = 1  # set their emotion to be 1
    leader['alpha'] = 0  # set their alpha to be 0
    del leader['Expressiveness']  # delete the intimacy (Expressiveness) parameter since it's not defined for the leader  (we could have also ignored it/do nothing since it will not be used)
    leader.update({'emotionManagementAbility': uniform(0,1)})  # add emotionManagementAbility parameter  - method corresponds to dictionary option above
    # The below needs to be updated because I would really like a more dynamic way of determing when the leader intervenes, but in the meantime it is a semi-fixed value   
    if leader['emotionManagementAbility'] >= 0.5:
        leader.update({'interventionThreshold': 0.7})
    else:
        leader.update({'interventionThreshold': 0.3})
    #leader.insert(4, random.uniform(0,1))  # add emotionManagementAbility parameter  - method corresponds to list option above


    ### Intimacy weights - current option: 1

    ## dictionary approach (more dynamic) 
    #intimacyDict = {agent_r: {agent_c: random.uniform(0,1) for agent_c in range(populationSize) if agent_c != agent_r} for agent_r in range(populationSize)}  # for each agent i create a connection to another agent j and randomly assign a value between 0 and 1

    ## matrix approach (faster and simpler computation)
    intimacyMatrix = np.random.uniform(0,1, (populationSize, populationSize))
    np.fill_diagonal(intimacyMatrix, 0)  # along the diagonal are the self-to-self pairs (agent_r, agent_r), so we may assign a weight of 0


    ### calculate expressiveness -- consider normalizing summation
    overallIntimacySummations = np.sum(intimacyMatrix, axis=1)  # sum row-wise (axis=1) how each agent_r feels towards other agents
    for r, agent in enumerate(agents):  # for each agent at index r
        agent['Expressiveness'] = overallIntimacySummations[r]  # assign it's respective sum which is stored correspondingly at index r in the summation list


    envir = zeros([height, width])  # initialize main environment (2D array of 0s with shape of our predefined dimensions)
    for y in range(height):  # for each vertical step
        for x in range(width):  # for each horizontal step (so in combination, for each cell)
            envir[y, x] = random()  # assign to that pair of coordinates a random integer (representing an agent?)

    nextenvir = zeros([height, width])  # initialize next environment


def observe():
    '''
    Creates the simulator for us to see.
    '''
    cla()  # clear plot
    imshow(envir, cmap = cm.YlOrRd, vmin = 0, vmax = 3)  # environment colors -- define color range (this creates sort of a heatmap)
    axis('image')  # proportional grid
    x = [ag['x'] for ag in agents]   # get x coordinate of each agent
    y = [ag['y'] for ag in agents]  # get y coordinate of each agent (together, this get's the coordinates of the agents)
    leader_x, leader_y = leader['x'], leader['y']  # get leader's coordinates
    
    scatter(x, y, cmap = cm.bone, alpha = 0.2)  # plot agents...
    scatter(leader_x, leader_y, s=100, color='black', marker='*', alpha = 0.6)  # plot leader...
    
    title('t = ' + str(time))  # show current step at time t as title


def clip(a, amin, amax):
    '''
    Ensure values stay within range so agents stay within the boundaries of the grid (so they don't just disapear).
    '''
    if a < amin: return amin  # if a coordinate of an agent is below some value, set it to amin
    elif a > amax: return amax  # if a coordinate of an agent is above some value, set it to amax
    else: return a  # otherwise, it can stay as is


# Time for the meat of the simulation, defining the interactions!

def emotional_valence_update(agent_r, agent_c):
    '''
    For updating the emotional valence of each agent when they interact with the other agents.
    '''
    agent_r['Emotion'] = agent_r['Emotion'] - agent_r['alpha']*(agent_r['Emotion'] - agent_c['Emotion'])  # the stronger emotion is the more influential one, WRT agent_r's susceptibilty  
    agent_c['Emotion'] = agent_c['Emotion'] - agent_c['alpha']*(agent_c['Emotion'] - agent_r['Emotion'])


## when to put in the calculation of emotion average?
def avgEmotion(agents):
    '''
    Calculate average emotion valence amongst the team.
    '''
    return (sum(agent['Emotion'] for agent in agents))/max(1, len(agents))  # the max(1,...) is there in case agents is 0 or empty


def update_agents():
    '''
    Define inter-agent interactions within the simulation. 
    If they're within a certain distance of each other, that is an interaction.
    '''
    # so i dont need this below line?
    #neighbors = [nb for nb in agents if nb != ag and (ag['x'] - nb['x'])**2 + (ag['y'] - nb['y'])**2 < crsq]

    global agents, avgEmotionValence

    avgEmotionValence = avgEmotion(agents)  # get the initial avg emotion    

    # agent movement -- for now it is random, though we may consider implementing hubs as mentioned above    
    for agent in agents:
        agent['x'] += randint(-1,2)
        agent['y'] += randint(-1,2)
        agent['x'] = clip(agent['x'], 0, width - 1)
        agent['y'] = clip(agent['y'], 0, height - 1)

        
    ## if at least two agents are within the same radius, they have potential to interact
    neighbor_pairs = []
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i != j:
                if (agent1['x'] - agent2['x'])**2 + (agent1['y'] - agent2['y'])**2 < crsq:
                        neighbor_pairs.append((i,j))  # use the indices rather than the dicts...

    # positions = np.array([(agent['x'], agent['y']) for agent in agents])
    # distances = np.linalg.norm(positions[:, None] - positions, axis=2)  # Compute pairwise distances
    # neighbor_pairs = np.argwhere((distances < cr) & (distances > 0))  # Get valid neighbors

    # neighbor_pairs = [(agent1, agent2) for i, agent1 in enumerate(agents) for j, agent2 in enumerate(agents) if i != j and (agent1['x'] - agent2['x'])**2 + (agent1['y'] - agent2['y'])**2 < crsq]
    
    # if the intimacy between the two (in either direction) is strong enough, we define this as an interaction
    if neighbor_pairs:
        i,j = choice(neighbor_pairs)
        agent1, agent2 = agents[i], agents[j]  # get the agent objects
        if max(intimacyMatrix[int(i),int(j)], intimacyMatrix[int(j),int(i)]) >= 0.5:
            emotional_valence_update(agent1, agent2)
        


def update():
    '''
    Update simulation. 
    Here we put in the rules, such as:
        Define how parameters/variables are to be updated (Emotion).
        When a leader intervenes and what happens when they do (we do not have to specify what the intervention is, just the degree that it impacts the team and here we may get creative).
    We may also consider the creation of "hubs" where employees are likely to interact (simulating areas such as kitchenette/eating area, watercooler, bathroom, desks). This way, their movement has some sense
    '''
    global time, agents, envir

    time += 1  # increment simulation time 
    max_iter = 200  # A limit just in case convergence takes too long. This will be fine-tuned as we run simulations; or we could use some sort of decay
    iteration = 0
    # while not (negConvergenceThreshold < avgEmotionValence < posConvergenceThreshold) and iteration < max_iter:
    #     update_agents()
    #     # should we have the leader intervene before or after updating the agents?
    #     if avgEmotionValence <= leader['interventionThreshold']:  # leader intervenes once the avg emotion reaches a certain predefined level -- HEY gotta be careful with this definition... ALso, is there a "failure" or do we just run until we decide to stop?
    #         for agent in agents:
    #             agent['Emotion'] += 0.1

    while iteration < max_iter:   
        update_agents()
        avgEmotionValence = avgEmotion(agents)
        if negConvergenceThreshold < avgEmotionValence < posConvergenceThreshold:
            break

        # leader intervenes once the avg emotion reaches a certain predefined level -- HEY gotta be careful with this definition... ALso, is there a "failure" or do we just run until we decide to stop?
        if avgEmotionValence <= leader['interventionThreshold']:  
            for agent in agents:
                agent['Emotion'] += 0.1
        
        iteration += 1


pycxsimulator.GUI().start(func=[initialize, observe, update])
