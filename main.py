from typing import Final
from DoRun import run
import matplotlib.pyplot as plt

NUMBER_OF_RUNS: Final = 2000
allAverageRewardsIncremental: list[list[float]] = []
allAverageRewardsWeighted: list[list[float]] = []

def getSingleListOfAverages(listContainingListsOfAverages: list[list[float]]):
    singleListOfAverages: list[float] = []
    for i in range(len(listContainingListsOfAverages[0])):
        average = 0
        for _,listOfAverages in enumerate(listContainingListsOfAverages):
            average += listOfAverages[i]
        average /= len(listContainingListsOfAverages)
        singleListOfAverages.append(average)
    return singleListOfAverages

for i in range(NUMBER_OF_RUNS):
    singleRun = run(useIncrementalEstimateCalculation=True)
    allAverageRewardsIncremental.append(singleRun['averageRewards'])
    print(str(round(((i + 1) / (NUMBER_OF_RUNS * 2)) * 100, 2)) + "%")

for i in range(NUMBER_OF_RUNS):
    singleRun = run(useIncrementalEstimateCalculation=False)
    allAverageRewardsWeighted.append(singleRun['averageRewards'])
    print(str(round(((i + 1 + NUMBER_OF_RUNS) / (NUMBER_OF_RUNS * 2)) * 100, 2)) + "%")

line, = plt.plot(getSingleListOfAverages(allAverageRewardsIncremental), 'r')
line.set_label('Incremental')

line, = plt.plot(getSingleListOfAverages(allAverageRewardsWeighted), 'b')
line.set_label('Weighted')

plt.ylabel("Average reward over " + str(NUMBER_OF_RUNS) + " runs")
plt.xlabel("Step")
plt.legend()

plt.suptitle("Incremental vs Weighted Estimate Calculations On Nonstationary Bandit Problem")

plt.show()
