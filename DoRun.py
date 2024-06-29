from collections.abc import Callable
from typing import Final, TypedDict
import random

# returns averageRewardOverTheLast100000Steps
def run(useIncrementalEstimateCalculation: bool, chanceToSelectRandomly: float):
    STEP_SIZE_PARAMETER: Final = 0.1

    NUMBER_OF_STEPS: Final = 2 * 10**6

    DEFAULT_ESTIMATE: Final = 0

    # Page 31 Second Edition Barto and Sutton
    def calculateNewAverageIncrementally(oldAverage, nextValue, numberOfValues):
        return oldAverage + (1/numberOfValues) * (nextValue - oldAverage)

    def calculateNewAverageWithStepSizeParameter(oldAverage, nextValue, stepSizeParameter):
        return oldAverage + (stepSizeParameter) * (nextValue - oldAverage)

    class Lever(TypedDict):
        estimate: None | float
        getReward: Callable[[], float]
        takeRandomWalk: Callable[[], None]
        getTrueValue: Callable[[], float]

    def createLever() -> Lever:
        trueValue = random.normalvariate(0, 1)
        def takeRandomWalk():
            nonlocal trueValue
            trueValue += random.normalvariate(0, 0.01)
        return {
            "estimate": None,
            "getReward": lambda: random.normalvariate(trueValue, 1),
            "takeRandomWalk": takeRandomWalk,
            "getTrueValue": lambda: trueValue
        }

    def getOptimalLever(levers: list[Lever]):
        optimalLever = levers[0]
        for lever in levers:
            if (optimalLever["getTrueValue"]() < lever["getTrueValue"]()):
                optimalLever = lever
        return optimalLever

    def chooseLeverRandomly():
        return random.choice(levers)

    def chooseLeverGreedily():
        def getHighestEstimateLevers(list: list[Lever]) -> list[Lever]:
            highestEstimate = -999
            highestEstimateLevers = []
            for _,lever in enumerate(list):
                estimate = lever['estimate'] if lever['estimate'] is not None else DEFAULT_ESTIMATE
                if (estimate > highestEstimate):
                    highestEstimateLevers = []
                    highestEstimate = estimate
                    highestEstimateLevers.append(lever)
                elif(estimate == highestEstimate):
                    highestEstimateLevers.append(lever)
            return highestEstimateLevers

        highestEstimateLevers = getHighestEstimateLevers(levers)
        return random.choice(highestEstimateLevers)

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

    optimalLever = getOptimalLever(levers)

    for i in range(NUMBER_OF_STEPS):
        def chooseLever():
            if (random.random() < chanceToSelectRandomly):
                return chooseLeverRandomly()
            else:
                return chooseLeverGreedily()

        def updateEstimate(lever, reward):
            if (lever['estimate'] is None):
                lever['estimate'] = reward
            else:
                if (useIncrementalEstimateCalculation):
                    lever['estimate'] = calculateNewAverageIncrementally(lever['estimate'], reward, i + 1)
                else:
                    lever['estimate'] = calculateNewAverageWithStepSizeParameter(lever['estimate'], reward, STEP_SIZE_PARAMETER)

        def updateAverageRewardOverTheLast100000Steps(reward):
            nonlocal averageRewardOverTheLast100000Steps
            # dividing by 2 incase I lower the NUMBER_OF_STEPS for testing
            if (i < (NUMBER_OF_STEPS / 2)):
                return

            if (i == (NUMBER_OF_STEPS / 2)):
                averageRewardOverTheLast100000Steps = reward
            else:
                averageRewardOverTheLast100000Steps = calculateNewAverageIncrementally(averageRewardOverTheLast100000Steps, reward, (i - 10**6) + 1)
                


        def walkLevers():
            if (ARE_LEVERS_WALKING):
                for _,lever in enumerate(levers):
                    lever["takeRandomWalk"]()
            for lever in levers:
                lever["takeRandomWalk"](i)

        lever = chooseLever()
        reward = lever['getReward']()

        updateEstimate(lever, reward)
        updateAverageRewardOverTheLast100000Steps(reward)
        walkLevers()

    return averageRewardOverTheLast100000Steps 

def multipleRuns(useIncrementalEstimateCalculation: bool, chanceToSelectRandomly: float, runs: int):
    averageRewards: list[float] = []
    for i in range(runs):
        averageRewards.append(run(useIncrementalEstimateCalculation, chanceToSelectRandomly))
        print(str(((i + 1) / runs) * 100) + str("%"))
    average = sum(averageRewards) / len(averageRewards)
    return average
