import os
import numpy as np
from typing import Final
from DoRun import getChooseActionUCB, multipleRuns, getChooseActionGreedy, getChooseActionGradient
import matplotlib.pyplot as plt
from concurrent import futures

PARAMETERS: Final = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
PARAMETERS_AS_STRING: Final = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]

NUMBER_OF_RUNS: Final = 2000
NUMBER_OF_STEPS: Final = 2 * 10**5

averageRewardsEpsilonGreedy: list[float] = []
averageRewardsGradient: list[float] = []
averageRewardsGreedyWithOptimisticInitialization: list[float] = []
averageRewardsUpperConfidenceBound: list[float] = []

with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
    def epsilonGreedy(chanceToSelectRandomly):
        return multipleRuns(
            lambda: getChooseActionGreedy(chanceToSelectRandomly),
            "Epsilon greedy: " + str(round(chanceToSelectRandomly, 2)),
            NUMBER_OF_RUNS,
            NUMBER_OF_STEPS
        )

    def greedyWithOptimisticInitialization(defaultEstimate):
        return multipleRuns(
            lambda: getChooseActionGreedy(
                chanceToSelectRandomly=0.1,
                defaultEstimate=defaultEstimate
            ),
            "Optimistic: " + str(round(defaultEstimate, 2)),
            NUMBER_OF_RUNS,
            NUMBER_OF_STEPS
        )

    def gradient(stepSizeParameter):
        return multipleRuns(
            lambda: getChooseActionGradient(stepSizeParameter),
            "Gradient " + str(round(stepSizeParameter, 2)),
            NUMBER_OF_RUNS,
            NUMBER_OF_STEPS
        )

    def UCB(degreeOfExploration):
        return multipleRuns(
            lambda: getChooseActionUCB(degreeOfExploration),
            "UCB " + str(round(degreeOfExploration ,2)),
            NUMBER_OF_RUNS,
            NUMBER_OF_STEPS
        )

    averageRewardsEpsilonGreedy = list(ex.map(epsilonGreedy, PARAMETERS))
    averageRewardsGreedyWithOptimisticInitialization = list(ex.map(greedyWithOptimisticInitialization, PARAMETERS))
    averageRewardsGradient = list(ex.map(gradient, PARAMETERS))
    averageRewardsUpperConfidenceBound = list(ex.map(UCB, PARAMETERS))

plt.plot(averageRewardsEpsilonGreedy, 'r')
plt.plot(averageRewardsGreedyWithOptimisticInitialization, 'k')
plt.plot(averageRewardsGradient, 'g')
plt.plot(averageRewardsUpperConfidenceBound, 'b')
plt.xticks(np.arange(len(PARAMETERS)), PARAMETERS_AS_STRING)

plt.ylabel("Average reward over the last 100,000 steps")
plt.xlabel("Parameters (Epsilon, alpha, c, optimistic estimate)")

plt.show()
