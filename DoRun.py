from collections.abc import Callable
from typing import Final, TypedDict
import random
import numpy as np
from numpy.lib import math

# returns averageRewardOverTheLast100000Steps
def runGreedy(useIncrementalEstimateCalculation: bool, chanceToSelectRandomly: float, defaultEstimate: float = 0):
    STEP_SIZE_PARAMETER: Final = 0.1

    NUMBER_OF_STEPS: Final = 2 * 10**6

    # Page 31 Second Edition Barto and Sutton
    def calculateNewAverageIncrementally(oldAverage, nextValue, numberOfValues):
        return oldAverage + (1/numberOfValues) * (nextValue - oldAverage)

    def calculateNewAverageWithStepSizeParameter(oldAverage, nextValue, stepSizeParameter):
        return oldAverage + (stepSizeParameter) * (nextValue - oldAverage)

    estimates: list[float | None] = [None for _ in range(10)]
    trueValues: list[float] = [random.normalvariate(0, 1) for _ in range(10)]

    def getReward(action: int):
        return random.normalvariate(trueValues[action], 1)

    def chooseLeverRandomly():
        return random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def chooseLeverGreedily():
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

    averageRewardOverTheLast100000Steps: float = 0

    for i in range(NUMBER_OF_STEPS):
        def chooseAction() -> int:
            if (random.random() < chanceToSelectRandomly):
                return chooseLeverRandomly()
            else:
                return chooseLeverGreedily()

        def updateEstimate(action, reward):
            if (estimates[action] is None):
                estimates[action] = reward
            else:
                if (useIncrementalEstimateCalculation):
                    estimates[action] = calculateNewAverageIncrementally(estimates[action], reward, i + 1)
                else:
                    estimates[action] = calculateNewAverageWithStepSizeParameter(estimates[action], reward, STEP_SIZE_PARAMETER)

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

        def walkLevers():
            for i,_ in enumerate(trueValues):
                trueValues[i] += getRandomWalkNumber()

        action = chooseAction()
        reward = getReward(action)

        updateEstimate(action, reward)
        updateAverageRewardOverTheLast100000Steps(reward)
        walkLevers()

    return averageRewardOverTheLast100000Steps 


def runGradient(stepSizeParameter: float):
    NUMBER_OF_STEPS: Final = 2 * 10**6

    averageReward: float = 0
    # Page 31 Second Edition Barto and Sutton
    def calculateNewAverageIncrementally(oldAverage, nextValue, numberOfValues):
        return oldAverage + (1/numberOfValues) * (nextValue - oldAverage)

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

def multipleRuns(useIncrementalEstimateCalculation: bool, chanceToSelectRandomly: float, runs: int, defaultEstimate = 0):
    averageRewards: list[float] = []
    for i in range(runs):
        averageRewards.append(runGreedy(useIncrementalEstimateCalculation, chanceToSelectRandomly, defaultEstimate))
        percentageComplete = str(((i + 1) / runs) * 100) + str("%")
        print("For ε=" + str(chanceToSelectRandomly) + " is " + percentageComplete)
    average = sum(averageRewards) / len(averageRewards)
    return average
