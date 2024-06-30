import os
import numpy as np
from typing import Final, TypedDict
from DoRun import multipleRuns
import matplotlib.pyplot as plt
from concurrent import futures

PARAMETERS: Final = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
PARAMETERS_AS_STRING: Final = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]

NUMBER_OF_RUNS: Final = 2000

averageRewardsEpsilonGreedy: list[float] = []
averageRewardsGradient: list[float] = []
averageRewardsGreedyWithOptimisticInitialization: list[float] = []
averageRewardsUpperConfidenceBound: list[float] = []

with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
    def epsilonGreedy(chanceToSelectRandomly):
        return multipleRuns(
            useIncrementalEstimateCalculation=False,
            chanceToSelectRandomly=chanceToSelectRandomly,
            runs=NUMBER_OF_RUNS
        )
    averageRewardsEpsilonGreedy = list(ex.map(epsilonGreedy, PARAMETERS))

def getResults(listOfFutures: list[futures.Future]):
    return [future.result() for future in listOfFutures]

plt.plot(averageRewardsEpsilonGreedy, 'r')
plt.xticks(np.arange(len(PARAMETERS)), PARAMETERS_AS_STRING)

plt.ylabel("Average reward over the last 100,000 steps")
plt.xlabel("Parameters (Epsilon, alpha, c, optimistic estimate)")

plt.show()
