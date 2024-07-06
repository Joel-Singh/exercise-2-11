from collections.abc import Callable
from typing import Final, TypedDict
import random
import numpy as np
from numpy.lib import math

# Page 31 Second Edition Barto and Sutton
def calculateNewAverageIncrementally(oldAverage: float, nextValue: float, numberOfValues: int):
    return oldAverage + (1/numberOfValues) * (nextValue - oldAverage)

def calculateNewAverageWithStepSizeParameter(oldAverage, nextValue, stepSizeParameter):
    return oldAverage + (stepSizeParameter) * (nextValue - oldAverage)

def getReward(action: int, trueValues: list[float]):
    return random.normalvariate(trueValues[action], 1)

ChooseAction = Callable[[int, list[float]], float]

# Returns reward
def getChooseActionGreedy(
    chanceToSelectRandomly: float = 0.1,
    defaultEstimate: float = 0
) -> ChooseAction:
    estimates: list[None | float] = [None for _ in range(10)]

    def chooseActionRandomly():
        return random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def chooseActionGreedily():
        def getHighestEstimateActions() -> list[int]:
            highestEstimate = -999
            highestEstimateActions = []
            for i in range(10):
                estimate = estimates[i] if estimates[i] is not None else defaultEstimate
                if (estimate > highestEstimate): # type:ignore
                    highestEstimateActions = []
                    highestEstimate = estimate
                    highestEstimateActions.append(i)
                elif(estimate == highestEstimate):
                    highestEstimateActions.append(i)
            return highestEstimateActions

        highestEstimateActions = getHighestEstimateActions()
        return random.choice(highestEstimateActions)

    STEP_SIZE_PARAMETER: Final = 0.1
    def updateEstimate(action: int, reward: float, currentStep: int):
        if (estimates[action] is None):
            estimates[action] = reward
        else:
            estimates[action] = calculateNewAverageWithStepSizeParameter(estimates[action], reward, STEP_SIZE_PARAMETER)

    def chooseAction(currentStep: int, trueValues: list[float]) -> float:
        action: int = 0
        if (random.random() < chanceToSelectRandomly):
            action = chooseActionRandomly()
        else:
            action = chooseActionGreedily()
        reward = getReward(action, trueValues)
        updateEstimate(action, reward, currentStep)
        return reward

    return chooseAction

def getChooseActionGradient(stepSizeParameter: float) -> ChooseAction:
    preferences: list[float] = [0 for _ in range(10)]
    averageReward = 0

    # Page 37
    def getProbabilities():
        total: float = 0
        for preference in preferences:
            total += math.e**(preference)

        return [math.e**preference / total for preference in preferences]

    def updateAverageReward(reward, currentStep):
        nonlocal averageReward
        averageReward = calculateNewAverageIncrementally(averageReward, reward, currentStep + 1)

    def updatePreferences(chosenAction: int, reward: float):
        probabilities = getProbabilities()
        # Page 37
        for i,preference in enumerate(preferences):
            if (i == chosenAction):
                preferences[i] = preference + stepSizeParameter * (reward - averageReward) * (1 - probabilities[i])
            else:
                preferences[i] = preference - stepSizeParameter * (reward - averageReward) * probabilities[i]

    def chooseAction(currentStep: int, trueValues: list[float]):
        action = random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], getProbabilities())[0] 
        reward: float = getReward(action, trueValues)
        updateAverageReward(reward, currentStep)
        updatePreferences(action, reward)
        return reward

    return chooseAction 

def getChooseActionUCB(degreeOfExploration: float) -> ChooseAction:
    estimates: list[float] = [0 for _ in range(10)]
    numberOfTimesChosen: list[int] = [0 for _ in range(10)]

    def updateEstimate(action: int, reward: float):
        estimates[action] = calculateNewAverageIncrementally(estimates[action], reward, numberOfTimesChosen[action])

    def chooseAction(currentStep: int, trueValues: list[float]):
        def getUCBEquationValue(estimate: float, numberOfTimes: int):
            # Page 35
            if (numberOfTimes == 0):
                return float('inf')
            return estimate + degreeOfExploration * math.sqrt(math.log(currentStep + 1) / numberOfTimes)

        def getHighestActions():
            UCBEquationValues = [getUCBEquationValue(estimates[a], numberOfTimesChosen[a]) for a in range(10)]
            highestActions: list[int] = []
            highestValue = float('-inf')
            for a,v in enumerate(UCBEquationValues):
                if (v < highestValue):
                    continue
                elif (v == highestValue):
                    highestActions.append(a)
                    continue
                elif (v > highestValue):
                    highestValue = v
                    highestActions.clear()
                    highestActions.append(a)

            return highestActions

        action: int = random.choice(getHighestActions())
        numberOfTimesChosen[action] += 1

        reward: float = getReward(action, trueValues)
        updateEstimate(action, reward)
        return reward

    return chooseAction

def run(chooseAction: ChooseAction, numberOfSteps):
    trueValues: list[float] = [random.normalvariate(0, 1) for _ in range(10)]

    averageRewardOverTheLast100000Steps: float = 0

    for i in range(numberOfSteps):
        def updateAverageRewardOverTheLast100000Steps(reward):
            nonlocal averageRewardOverTheLast100000Steps
            # dividing by 2 incase I lower the NUMBER_OF_STEPS for testing
            if (i < (numberOfSteps / 2)):
                return

            if (i == (numberOfSteps / 2)):
                averageRewardOverTheLast100000Steps = reward
            else:
                averageRewardOverTheLast100000Steps = calculateNewAverageIncrementally(averageRewardOverTheLast100000Steps, reward, (i - 10**6) + 1)
                
        currentRandomNumber: int = -1
        randomWalkNumbers: np.ndarray = np.random.normal(0, 0.01, numberOfSteps * 10)
        def getRandomWalkNumber():
            nonlocal currentRandomNumber
            nonlocal randomWalkNumbers

            currentRandomNumber = currentRandomNumber + 1
            return randomWalkNumbers[currentRandomNumber]

        def walkActions():
            for i,_ in enumerate(trueValues):
                trueValues[i] += getRandomWalkNumber()

        reward = chooseAction(i, trueValues)

        updateAverageRewardOverTheLast100000Steps(reward)
        walkActions()

    return averageRewardOverTheLast100000Steps 


def multipleRuns(chooseActionGetter: Callable[[], ChooseAction], name: str, runs: int, numberOfSteps: int):
    averageRewards: list[float] = []
    for i in range(runs):
        averageReward = run(chooseActionGetter(), numberOfSteps)
        averageRewards.append(averageReward)
        percentageComplete = str(((i + 1) / runs) * 100) + str("%")
        print("For " + name + " is " + percentageComplete)
    average = sum(averageRewards) / len(averageRewards)
    return average
