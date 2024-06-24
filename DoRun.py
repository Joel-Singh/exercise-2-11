from collections.abc import Callable
from typing import Final, TypedDict
import random

class Run(TypedDict):
    percentageOfOptimalLeverChosen: list[float]
    averageRewardOverTheLast100000Steps: float

def run(useIncrementalEstimateCalculation: bool, chanceToSelectRandomly: float) -> Run:
    STEP_SIZE_PARAMETER: Final = 0.1

    NUMBER_OF_STEPS: Final = 2 * 10**6

    ARE_LEVERS_WALKING: Final = True

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
    percentageOfOptimalLeverChosen: list[float] = []

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
            if (i < 10**6):
                return

            if (i == 10**6):
                averageRewardOverTheLast100000Steps = reward
            else:
                averageRewardOverTheLast100000Steps = calculateNewAverageIncrementally(averageRewardOverTheLast100000Steps, reward, (i - 10**6) + 1)
                

        def updatePercentageOfOptimalLeverChosen():
            nonlocal percentageOfOptimalLeverChosen
            leverChosenWasOptimalAsInt = 1 if lever is optimalLever else 0
            if (i == 0):
                percentageOfOptimalLeverChosen.append(leverChosenWasOptimalAsInt)
            else:
                percentageOfOptimalLeverChosen.append(
                    calculateNewAverageIncrementally(percentageOfOptimalLeverChosen[i - 1], leverChosenWasOptimalAsInt, i + 1)
                )

        def walkLevers():
            if (ARE_LEVERS_WALKING):
                for _,lever in enumerate(levers):
                    lever["takeRandomWalk"]()

        lever = chooseLever()
        reward = lever['getReward']()

        updatePercentageOfOptimalLeverChosen()

        updateEstimate(lever, reward)
        updateAverageRewardOverTheLast100000Steps(reward)
        walkLevers()

    return {
        "percentageOfOptimalLeverChosen": percentageOfOptimalLeverChosen,
        "averageRewardOverTheLast100000Steps": averageRewardOverTheLast100000Steps
    }
