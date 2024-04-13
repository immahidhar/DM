import numpy
import random
from sklearn import preprocessing

from Sample import Sample


# scale data set - mean 0 and variance 1
def scaleData(dataSet):
    scaler = preprocessing.StandardScaler().fit(dataSet)
    dataSet_scaled = scaler.transform(dataSet)
    return dataSet_scaled


# get accuracy of predicted vs actual
def getAccuracy(predicted, actual):
    correct = (predicted == actual)
    accuracy = correct.sum() / correct.size
    return accuracy


# pick k random samples from uSet and corresponding uLabels and append to tSet and tLabels
def getRandomKSamplesFromHuman(tSet, uSet, tLabels, uLabels, k):
    uSet = numpy.array(uSet)
    for i in range(k):
        nRows, nCols = uSet.shape
        randomIndex = random.randrange(nRows)
        row = uSet[randomIndex]
        # get label - simulating human expertise
        label = uLabels[randomIndex]
        uSet = numpy.delete(uSet, randomIndex, axis=0)
        tSet = numpy.vstack([tSet, row])
        uLabels = numpy.delete(uLabels, randomIndex, axis=0)
        tLabels = numpy.append(tLabels, label, axis=0)
    return tSet, uSet, tLabels, uLabels


# pick top k uncertain samples from uSet and corresponding uLabels and append to tSet and tLabels
def getUncertainKSamplesFromHuman(trSet, uSet, trLabels, uLabels, lr, k):
    # predict labels of unlabeled set to calculate entropy
    uSetSamples = list()
    nRows, nCols = uSet.shape
    p = lr.predict_proba(uSet)
    for i in range(nRows):
        sample = Sample(i, uSet[i], None, p[i])
        uSetSamples.append(sample)

    # sort unlabeled samples based on entropy in descending order
    uSetSamples.sort(key=lambda s: s.entropy, reverse=True)

    # get indices of the k highest entropy samples
    indexList = list()
    for i in range(k):
        indexList.append(uSetSamples[i].index)

    # move samples with indices in indexList from uSet to trSet
    # sort indexList in descending order as we are removing inside loop
    indexList.sort(reverse=True)
    for index in indexList:
        row = uSet[index]
        # get label - simulating human expertise
        label = uLabels[index]
        uSet = numpy.delete(uSet, index, axis=0)
        trSet = numpy.vstack([trSet, row])
        uLabels = numpy.delete(uLabels, index, axis=0)
        trLabels = numpy.append(trLabels, label, axis=0)
    return trSet, uSet, trLabels, uLabels
