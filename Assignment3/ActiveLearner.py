import numpy
from enum import Enum
from sklearn import linear_model

from ALUtil import scaleData, getRandomKSamplesFromHuman, getUncertainKSamplesFromHuman, getAccuracy


class ALStrategy(Enum):
    RANDOM = 1
    UNCERTAINTY = 2


# active learning based on a given strategy
def activeLearn(trainingSets, trainingLabelsSet, unlabeledSets, unlabeledLabelsSet, testingSets,
                testingLabelsSet, numberOfTestSets, N, k, strategy, scalingEnabled, maxIter):
    accuracies = list()
    iterations = list()

    for testSetNumber in range(numberOfTestSets):

        # get training set and scale it
        trainingSet = trainingSets.__getitem__(testSetNumber)
        if scalingEnabled:
            trainingSet = scaleData(trainingSet)
        # labels of trainingSet
        trainingLabels = trainingLabelsSet.__getitem__(testSetNumber)
        trainingLabels = numpy.array(trainingLabels).reshape(-1)

        # get unlabeled sets and scale it
        unlabeledSet = unlabeledSets.__getitem__(testSetNumber)
        if scalingEnabled:
            unlabeledSet = scaleData(unlabeledSet)
        unlabeledLabels = unlabeledLabelsSet.__getitem__(testSetNumber)

        # get test set and scale it
        testSet = testingSets.__getitem__(testSetNumber)
        if scalingEnabled:
            testSet = scaleData(testSet)
        actual = numpy.array(testingLabelsSet.__getitem__(testSetNumber)).reshape(-1)

        accuracyList = list()
        iterationList = list()

        for iteration in range(N + 1):
            # train Logistic Regression model
            if scalingEnabled:
                logr = linear_model.LogisticRegression()
            else:
                logr = linear_model.LogisticRegression(max_iter=maxIter)
            logr.fit(trainingSet, trainingLabels)

            # predict labels of testSet
            predicted = logr.predict(testSet)

            # calculate accuracy
            accuracy = getAccuracy(predicted, actual)

            # store data for plots
            accuracyList.append(accuracy)
            iterationList.append(iteration)

            # get k uncertain samples by simulating human expertise
            if strategy == ALStrategy.RANDOM:
                trainingSet, unlabeledSet, trainingLabels, unlabeledLabels = \
                    getRandomKSamplesFromHuman(trainingSet, unlabeledSet, trainingLabels, unlabeledLabels, k)
            if strategy == ALStrategy.UNCERTAINTY:
                trainingSet, unlabeledSet, trainingLabels, unlabeledLabels = \
                    getUncertainKSamplesFromHuman(trainingSet, unlabeledSet, trainingLabels, unlabeledLabels, logr, k)

        iterations.append(iterationList)
        accuracies.append(accuracyList)

    # get average accuracies
    accuracies = [((accuracies[0][x] + accuracies[1][x] + accuracies[2][x]) / 3) for x in range(len(accuracies[0]))]

    return accuracies, iterations[0]
