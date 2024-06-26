import os
from typing import Final
from DoRun import run
import matplotlib.pyplot as plt
from concurrent import futures

PARAMETERS: Final = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]

averageRewardsOverTheLast100000StepsAsFutures: list[futures.Future] = []

with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    def runAndGetAverageReward(index):
        return run(
            useIncrementalEstimateCalculation=False,
            chanceToSelectRandomly=PARAMETERS[index]
        )['averageRewardOverTheLast100000Steps']

    for i,_ in enumerate(PARAMETERS):
        averageRewardsOverTheLast100000StepsAsFutures.append(ex.submit(runAndGetAverageReward, i))

plt.plot(PARAMETERS, [future.result() for future in averageRewardsOverTheLast100000StepsAsFutures], 'r')

plt.ylabel("Average reward over the last 100,000 steps")
plt.xlabel("Parameters (Epsilon, alpha, c, optimistic estimate)")

plt.show()
