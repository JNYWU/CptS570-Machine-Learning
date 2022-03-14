#%%

import numpy as np
import random
from math import exp

from numpy.random.mtrand import random_sample

gridSize = 10, 10
walls = [(2,1), (2,2), (2,3), (2,4), (2,6), (2,7), (2,8), (3,4), (4,4), (5,4), (6,4), (7,4)]
negativeRewards = [(3,3), (4,5), (4,6), (5,6), (5,8), (6,8), (7,3), (7,5), (7,6)]
positiveRewards = [(5,5)]

# create grid world
def GridWorld(gridSize, walls, negativeRewards, positiveRewards):
    gridWorld = np.zeros(shape = gridSize)

    # let walls be 99
    for i in walls:
        gridWorld[i] = 99

    for i in negativeRewards:
        gridWorld[i] = -1

    for i in positiveRewards:
        gridWorld[i] = 1

    return gridWorld

def PositionalGridWorld(gridWorld):
# represent grid world in a positional way
#  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
# 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#                   ...
# 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

    i = 0
    positionalGridWorld = np.zeros(shape = gridWorld.shape, dtype= int)

    for row in range(gridWorld.shape[0]):
        for col in range(gridWorld.shape[1]):
            positionalGridWorld[row][col] = i
            i += 1
    
    return positionalGridWorld

def PossibleActions(currentState, gridWorld):
    # 0 = up, 1 = down, 2 = left, 3 = right

    currentX = currentState[0]
    currentY = currentState[1]

    possibleActions = []
    if currentX-1 >= 0 and gridWorld[currentX-1][currentY] != 99:
        possibleActions.append(0)

    if currentX+1 < 10 and gridWorld[currentX+1][currentY] != 99:
        possibleActions.append(1)

    if currentY-1 >= 0 and gridWorld[currentX][currentY-1] != 99:
        possibleActions.append(2)

    if currentY+1 < 10 and gridWorld[currentX][currentY+1] != 99:
        possibleActions.append(3)

    return possibleActions


def EpsilonGreedy(gridWorld, positionalGridWorld, epsilon):
    stepCount = 0
    currentState = (0,0)
    reward = gridWorld[currentState]

    # 100 grids * 4 actions
    Q = np.zeros(shape = (100,4))

    while reward != 1.0:
        reward = gridWorld[currentState]
        possibleActions = PossibleActions(currentState, gridWorld)
        currentStateIndex = positionalGridWorld[currentState]

        maxExploit = 0
        maxQ = 0
        actionTake = []
        nextState = (0,0)

        if len(possibleActions) > 0:
            # exploit
            if random.uniform(0,1) <= epsilon:
                for action in possibleActions:
                    # select action
                    if maxExploit <= Q[currentStateIndex][action]:
                        actionTake = action
                        maxExploit = Q[currentStateIndex][action]

            # explore
            else:
                actionTake = random.sample(possibleActions, 1)
                actionTake = actionTake[0]

            # take action
            if actionTake == 0:   # up
                nextState = (currentState[0]-1, currentState[1])
            elif actionTake == 1: # down
                nextState = (currentState[0]+1, currentState[1])
            elif actionTake == 2: # left
                nextState = (currentState[0], currentState[1]-1)
            elif actionTake == 3: # right
                nextState = (currentState[0], currentState[1]+1)

            nextStateIndex = positionalGridWorld[nextState]
            nextPossibleActions = PossibleActions(nextState, gridWorld)

            for action in nextPossibleActions:
                if maxQ <= Q[nextStateIndex][action]:
                    maxQ = Q[nextStateIndex][action]

            # bellman equation
            alpha = 0.01
            beta = 0.9
            Q[currentStateIndex][action] = Q[currentStateIndex][action] + alpha * (reward + (beta * maxQ) - Q[currentStateIndex][action])

            stepCount += 1
            currentState = nextState

    return stepCount, Q

def BoltzmanProbability(Q, currentStateIndex, possibleActions, stepCount, temperature):
    if stepCount > 0 and stepCount%10 == 0 and temperature != 0:
        temperature -= 0.05
        
    actionProbabilities = []
    
    denominator = 0
    
    for action in possibleActions:
        q = Q[currentStateIndex][action]
        denominator += exp(q/temperature)
    
    for action in possibleActions:
        q = Q[currentStateIndex][action]
        numerator = exp(q/temperature)
        
        if denominator != 0:
            probability = numerator / denominator
        else:
            probability = 0.0
        
        actionProbabilities.append(probability)
    
    return actionProbabilities, temperature
        

def BoltzmanExploration(gridWorld, positionalGridWorld):
    stepCount = 0
    currentState = (0,0)
    reward = gridWorld[currentState]
    temperature = 10
    allTemperatures = []
        
    # 100 grids * 4 actions
    Q = np.zeros(shape = (100,4))
    
    while reward != 1.0:
        reward = gridWorld[currentState]
        possibleActions = PossibleActions(currentState, gridWorld)
        currentStateIndex = positionalGridWorld[currentState]

        actionTake = []
        nextState = (0,0)
        maxQ = 0
        
        if len(possibleActions) > 0:
            # calculate probability
            print(possibleActions)
            actionProbabilities, temperature = BoltzmanProbability(Q, currentStateIndex, possibleActions, stepCount, temperature)
            maxProbability = max(actionProbabilities)
            minProbability = min(actionProbabilities)
            
            # scheduling rate 0.001
            if maxProbability - minProbability <= 0.001:
                actionTake = random.sample(possibleActions, 1)
                actionTake = actionTake[0]
            else:
                actionTake = actionProbabilities.index(maxProbability)
                
            # take action
            if actionTake == 0:   # up
                nextState = (currentState[0]-1, currentState[1])
            elif actionTake == 1: # down
                nextState = (currentState[0]+1, currentState[1])
            elif actionTake == 2: # left
                nextState = (currentState[0], currentState[1]-1)
            elif actionTake == 3: # right
                nextState = (currentState[0], currentState[1]+1)
                
            nextStateIndex = positionalGridWorld[nextState]
            nextPossibleActions = PossibleActions(nextState, gridWorld)
            
            for action in nextPossibleActions:
                if maxQ <= Q[nextStateIndex][action]:
                    maxQ = Q[nextStateIndex][action]
            
            # bellman equation
            alpha = 0.01
            beta = 0.9
            Q[currentStateIndex][action] = Q[currentStateIndex][action] + alpha * (reward + (beta * maxQ) - Q[currentStateIndex][action])
            stepCount += 1
            currentState = nextState
            allTemperatures.append(temperature)
            
    return stepCount, Q, allTemperatures

if __name__ == "__main__":
    gridWorld = GridWorld(gridSize, walls, negativeRewards, positiveRewards)
    positionalGridWorld = PositionalGridWorld(gridWorld)

    output = open("output1.txt", "a+")

    # epsilon greedy with epsilon 0.1, 0.2, 0.3
    print("1. 1) Epsilon Greedy", file = output)
    
    for i in range(1, 4):
        epsilon = i / 10

        stepCount, Q = EpsilonGreedy(gridWorld, positionalGridWorld, epsilon)

        print("epsilon =", epsilon, file = output)
        print("Iterations Taken :", stepCount, file = output)
        print("Q values", file = output)
        print(Q, file = output)

    output2 = open("output2.txt", "a+")

    # Boltzman Exploration starting with temperature = 10
    print("1. 2) Boltzman Exploration", file = output2)

    stepCount, Q, temperature = BoltzmanExploration(gridWorld, positionalGridWorld)
    
    print("Iterations Taken :", stepCount, file = output2)
    print("Q values", file = output2)
    print(Q, file = output2)

# %%
