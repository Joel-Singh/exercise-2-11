import os
import numpy as np
from typing import Final, TypedDict
from DoRun import run
import matplotlib.pyplot as plt
from concurrent import futures

PARAMETERS: Final = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
PARAMETERS_AS_STRING: Final = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]

class banditAlgorithmAverageRewardsAsFutures(TypedDict):
    epsilonGreedy: list[futures.Future]
    gradient: list[futures.Future]
    greedyWithOptimisticInitialization: list[futures.Future]
    upperConfidenceBound: list[futures.Future]

averageRewardsAsFutures: banditAlgorithmAverageRewardsAsFutures = {
    "epsilonGreedy": [],
    "gradient": [],
    "greedyWithOptimisticInitialization": [],
    "upperConfidenceBound": [],
}

with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    def epsilonGreedy(index):
        return run(
            useIncrementalEstimateCalculation=False,
            chanceToSelectRandomly=PARAMETERS[index]
        )['averageRewardOverTheLast100000Steps']

    for i,_ in enumerate(PARAMETERS):
        averageRewardsAsFutures["epsilonGreedy"].append(ex.submit(epsilonGreedy, i))

plt.plot([future.result() for future in averageRewardsAsFutures["epsilonGreedy"]], 'r')
plt.xticks(np.arange(len(PARAMETERS)), PARAMETERS_AS_STRING)

plt.ylabel("Average reward over the last 100,000 steps")
plt.xlabel("Parameters (Epsilon, alpha, c, optimistic estimate)")

plt.show()
