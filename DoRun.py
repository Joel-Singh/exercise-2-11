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

# Returns reward
def getChooseActionGreedy(useIncrementalEstimateCalculation: bool = False, chanceToSelectRandomly: float = 0.1, defaultEstimate: float = 0.1) -> Callable[[int, list[float]], float]:
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

    def getReward(action: int, trueValues: list[float]):
        return random.normalvariate(trueValues[action], 1)

    STEP_SIZE_PARAMETER: Final = 0.1
    def updateEstimate(action: int, reward: float, currentStep: int):
        if (estimates[action] is None):
            estimates[action] = reward
        else:
            if (useIncrementalEstimateCalculation):
                estimates[action] = calculateNewAverageIncrementally(estimates[action], reward, currentStep + 1)
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

def runGradient(stepSizeParameter: float):
    NUMBER_OF_STEPS: Final = 2 * 10**6

    averageReward: float = 0

    class Lever(TypedDict):
        preference: float
        getReward: Callable[[], float]
        takeRandomWalk: Callable[[], None]
        getTrueValue: Callable[[], float]

    currentRandomNumber: int = -1
    randomWalkNumbers: np.ndarray = np.random.normal(0, 0.01, NUMBER_OF_STEPS * 10)
    def getRandomWalkNumber():
        nonlocal currentRandomNumber
        nonlocal randomWalkNumbers

        currentRandomNumber = currentRandomNumber + 1
        return randomWalkNumbers[currentRandomNumber]

    def createLever() -> Lever:
        trueValue = random.normalvariate(0, 1)
        def takeRandomWalk():
            nonlocal trueValue
            trueValue += getRandomWalkNumber()
        return {
            "preference": 0,
            "getReward": lambda: random.normalvariate(trueValue, 1),
            "takeRandomWalk": takeRandomWalk,
            "getTrueValue": lambda: trueValue
        }

    levers = [
        createLever(),
        createLever(),
        createLever(),
        createLever(),
        createLever(),
        createLever(),
        createLever(),
        createLever(),
        createLever(),
        createLever(),
    ]

    averageRewardOverTheLast100000Steps: float = 0

    for i in range(NUMBER_OF_STEPS):
        # Page 37
        def getProbabilities():
            total: float = 0
            for lever in levers:
                total += math.e**(lever["preference"])

            return [math.e**lever["preference"] / total for lever in levers]

        def chooseLever():
            return random.choices(levers, getProbabilities())[0]

        def updatePreferences(chosenLever: Lever, reward: float):
            probabilities = getProbabilities()
            # Page 37
            for index,lever in enumerate(levers):
                if (lever is chosenLever):
                    lever["preference"] = lever["preference"] + stepSizeParameter * (reward - averageReward) * (1 - probabilities[index])

                
        def updateAverageRewardOverTheLast100000Steps(reward):
            nonlocal averageRewardOverTheLast100000Steps
            # dividing by 2 incase I lower the NUMBER_OF_STEPS for testing
            if (i < (NUMBER_OF_STEPS / 2)):
                return

            if (i == (NUMBER_OF_STEPS / 2)):
                averageRewardOverTheLast100000Steps = reward
            else:
                averageRewardOverTheLast100000Steps = calculateNewAverageIncrementally(averageRewardOverTheLast100000Steps, reward, (i - 10**6) + 1)

        def updateAverageReward(reward):
            nonlocal averageReward
            averageReward = calculateNewAverageIncrementally(averageReward, reward, i + 1)

        def walkLevers():
            for lever in levers:
                lever["takeRandomWalk"]()

        lever = chooseLever()
        reward = lever['getReward']()

        updatePreferences(lever, reward)
        updateAverageReward(reward)
        updateAverageRewardOverTheLast100000Steps(reward)
        walkLevers()

    return averageRewardOverTheLast100000Steps 

def run(chooseAction: Callable[[int, list[float]], float]):
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


def multipleRuns(useIncrementalEstimateCalculation: bool, chanceToSelectRandomly: float, runs: int, defaultEstimate = 0):
    averageRewards: list[float] = []
    for i in range(runs):
        averageReward = run(getChooseActionGreedy(
            useIncrementalEstimateCalculation,
            chanceToSelectRandomly,
            defaultEstimate
        ))
        averageRewards.append(averageReward)
        percentageComplete = str(((i + 1) / runs) * 100) + str("%")
        print("For ε=" + str(chanceToSelectRandomly) + " is " + percentageComplete)
    average = sum(averageRewards) / len(averageRewards)
    return average
