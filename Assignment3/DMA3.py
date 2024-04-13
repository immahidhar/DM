# Data Mining - Assignment 3 - MN22L
# References - https://www.w3schools.com/python/python_ml_logistic_regression.asp

from matplotlib import pyplot

from ActiveLearner import activeLearn, ALStrategy
from Loader import loadMMIDataSets, loadMindReadingDataSets

# define active learning parameters
k = 10
N = 50
numberOfTestSets = 3

enableScaling = True
maxIter = 4500

# load mmi data sets
mmiTrainingSets, mmiTrainingLabels, mmiUnlabeledSets, mmiUnlabeledLabels, mmiTestingSets, mmiTestingLabels = \
    loadMMIDataSets()

# load mind reading data sets
mrTrainingSets, mrTrainingLabels, mrUnlabeledSets, mrUnlabeledLabels, mrTestingSets, mrTestingLabels = \
    loadMindReadingDataSets()

# active learn mmi data set; strategy = random sampling
accuracies, iterations = activeLearn(mmiTrainingSets, mmiTrainingLabels, mmiUnlabeledSets, mmiUnlabeledLabels,
                                     mmiTestingSets, mmiTestingLabels, numberOfTestSets, N, k, ALStrategy.RANDOM,
                                     enableScaling, maxIter)
# plot random strategy accuracy curve
fig1 = pyplot.figure("Figure 1")
pyplot.plot(iterations, accuracies, label="random")

# active learn mmi data set; strategy = uncertainty-based sampling
accuracies, iterations = activeLearn(mmiTrainingSets, mmiTrainingLabels, mmiUnlabeledSets, mmiUnlabeledLabels,
                                     mmiTestingSets, mmiTestingLabels, numberOfTestSets, N, k, ALStrategy.UNCERTAINTY,
                                     enableScaling, maxIter)
# plot uncertain strategy accuracy curve
pyplot.plot(iterations, accuracies, label="uncertainty-based")
pyplot.title("MMI Dataset Active Learning Strategies")
pyplot.xlabel("Number of iterations")
pyplot.ylabel("Accuracy of model")
pyplot.grid(linestyle='--', linewidth=0.5)
pyplot.legend(loc="lower right")

# active learn mind reading data set; strategy = random sampling
accuracies, iterations = activeLearn(mrTrainingSets, mrTrainingLabels, mrUnlabeledSets, mrUnlabeledLabels,
                                     mrTestingSets, mrTestingLabels, numberOfTestSets, N, k, ALStrategy.RANDOM,
                                     enableScaling, maxIter)
# plot random strategy accuracy curve
fig2 = pyplot.figure("Figure 2")
pyplot.plot(iterations, accuracies, label="random")

# active learn mind reading data set; strategy = uncertainty-based sampling
accuracies, iterations = activeLearn(mrTrainingSets, mrTrainingLabels, mrUnlabeledSets, mrUnlabeledLabels,
                                     mrTestingSets, mrTestingLabels, numberOfTestSets, N, k, ALStrategy.UNCERTAINTY,
                                     enableScaling, maxIter)
# plot uncertain strategy accuracy curve
pyplot.plot(iterations, accuracies, label="uncertainty-based")
pyplot.title("Mind Reading Dataset Active Learning Strategies")
pyplot.xlabel("Number of iterations")
pyplot.ylabel("Accuracy of model")
pyplot.grid(linestyle='--', linewidth=0.5)
pyplot.legend(loc="lower right")

pyplot.show()
