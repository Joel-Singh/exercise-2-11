from typing import Final
from DoRun import run
import matplotlib.pyplot as plt

averageRewardsOverTheLast100000Steps: list[float] = []

PARAMETERS: Final = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
for index,parameter in enumerate(PARAMETERS):
    averageRewardsOverTheLast100000Steps.append(
        run(
            useIncrementalEstimateCalculation=False,
            chanceToSelectRandomly=parameter
        )['averageRewardOverTheLast100000Steps']
    )
    print(str(round((index + 1) / len(PARAMETERS), 2) * 100) + "%")

plt.plot(PARAMETERS, averageRewardsOverTheLast100000Steps, 'r')

plt.ylabel("Average reward over the last 100,000 steps")
plt.xlabel("Parameters (Epsilon, alpha, c, optimistic estimate)")

plt.show()
