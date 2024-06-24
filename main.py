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

firstRun = run(useIncrementalEstimateCalculation=True, chanceToSelectRandomly=0.1)
secondRun = run(useIncrementalEstimateCalculation=True, chanceToSelectRandomly=0.1)

print("averageRewardOverTheLast100000Steps is " + str(firstRun["averageRewardOverTheLast100000Steps"]))
plt.plot([firstRun['averageRewardOverTheLast100000Steps'], secondRun['averageRewardOverTheLast100000Steps']], 'r')

plt.ylabel("Average reward over the last 100,000 steps")
plt.xlabel("Parameters (Epsilon, alpha, c, optimistic estimate)")

plt.show()
