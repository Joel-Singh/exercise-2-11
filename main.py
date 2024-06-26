from typing import Final
from DoRun import run
import matplotlib.pyplot as plt
from concurrent import futures

PARAMETERS: Final = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]

averageRewardsOverTheLast100000Steps: list[float] = []
for _,_ in enumerate(PARAMETERS):
    averageRewardsOverTheLast100000Steps.append(0)

with futures.ProcessPoolExecutor(max_workers=4) as ex:
    def runAndInsertAverageReward(index, parameter):
        averageRewardsOverTheLast100000Steps[index] = run(
                useIncrementalEstimateCalculation=False,
                chanceToSelectRandomly=parameter
            )['averageRewardOverTheLast100000Steps']

        print("done")
    for index,parameter in enumerate(PARAMETERS):
        ex.submit(runAndInsertAverageReward, index, parameter)

plt.plot(PARAMETERS, averageRewardsOverTheLast100000Steps, 'r')

plt.ylabel("Average reward over the last 100,000 steps")
plt.xlabel("Parameters (Epsilon, alpha, c, optimistic estimate)")

plt.show()
