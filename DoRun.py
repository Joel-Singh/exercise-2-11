from collections.abc import Callable
from typing import Final, TypedDict
import random
import numpy as np
from numpy.lib import math

# Page 31 Second Edition Barto and Sutton
def calculateNewAverageIncrementally(oldAverage, nextValue, numberOfValues):
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

def run(chooseAction: ChooseAction):
    NUMBER_OF_STEPS: Final = 2 * 10**6

    trueValues: list[float] = [random.normalvariate(0, 1) for _ in range(10)]

    averageRewardOverTheLast100000Steps: float = 0

    for i in range(NUMBER_OF_STEPS):
        def updateAverageRewardOverTheLast100000Steps(reward):
            nonlocal averageRewardOverTheLast100000Steps
            # dividing by 2 incase I lower the NUMBER_OF_STEPS for testing
            if (i < (NUMBER_OF_STEPS / 2)):
                return

            if (i == (NUMBER_OF_STEPS / 2)):
                averageRewardOverTheLast100000Steps = reward
            else:
                averageRewardOverTheLast100000Steps = calculateNewAverageIncrementally(averageRewardOverTheLast100000Steps, reward, (i - 10**6) + 1)
                
        currentRandomNumber: int = -1
        randomWalkNumbers: np.ndarray = np.random.normal(0, 0.01, NUMBER_OF_STEPS * 10)
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


def multipleRuns(chooseActionGetter: Callable[[], ChooseAction], runs: int, name: str):
    averageRewards: list[float] = []
    for i in range(runs):
        averageReward = run(chooseActionGetter())
        averageRewards.append(averageReward)
        percentageComplete = str(((i + 1) / runs) * 100) + str("%")
        print("For " + name + " is " + percentageComplete)
    average = sum(averageRewards) / len(averageRewards)
    return average
