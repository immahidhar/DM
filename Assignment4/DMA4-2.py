import copy
import pandas
from math import sqrt
from numpy import mean

from loader import loadProblem2DataSet

# load dataset
dataSet = loadProblem2DataSet()
# print("\nDataSet:\n", dataSet)

# constants
num_iter_k = 10
max_iter = 100
min_sse_delta = 0.001
closestCentroidColumnName = "centroid"


def getEuclideanDistance(a, b):
    if a.shape == b.shape:
        n = a.size
        s = 0
        for i in range(n):
            s = s + ((a.iloc[i] - b.iloc[i]) ** 2)
        return sqrt(s)


def getSSE(distanceDF):
    distanceSquaredSum = 0
    for distanceIndex, distanceRow in distanceDF.iterrows():
        centroid = distanceRow[closestCentroidColumnName]
        distance = distanceRow[centroid]
        distanceSquared = distance ** 2
        distanceSquaredSum = distanceSquaredSum + distanceSquared
    return distanceSquaredSum


def kmeans(k):
    # 1. Randomly assign a centroid to each of the k clusters
    # just pick k random points out of the dataset as centroids
    dataFrame = copy.deepcopy(dataSet)
    centroidsDF = dataFrame.sample(k)
    # print("\nInitial centroids:\n", centroids)

    iter_count = 0
    lastSSE = 0
    sse = 0

    while (lastSSE == 0 or abs(sse - lastSSE) > min_sse_delta) and iter_count <= max_iter:

        lastSSE = sse

        # 2. Calculate the distance of all points to each of the k centroids
        distances = list()
        for dataIndex, dataRow in dataFrame.iterrows():
            centroidDistances = list()
            for centroidIndex, centroidRow in centroidsDF.iterrows():
                distance = getEuclideanDistance(dataRow, centroidRow)
                # print(dataIndex, centroidIndex, distance)
                centroidDistances.append(distance)
            distances.append(centroidDistances)
        distanceDF = pandas.DataFrame(distances)
        # print(distanceDF)

        # 3. Assign points to the closest centroid
        centroidIndices = list()
        for distanceIndex, distanceRow in distanceDF.iterrows():
            closestCentroidIndex = pandas.Index(distanceRow).get_loc(distanceRow.min())
            centroidIndices.append(closestCentroidIndex)
        distanceDF[closestCentroidColumnName] = centroidIndices
        # print(distanceDF)

        # 4. Form new centroids by taking the mean of all the points in each cluster
        centroids = list()
        for i in range(k):
            distanceDFCluster = distanceDF[distanceDF[closestCentroidColumnName] == i]
            # print(distanceDFCluster)
            dataFrameCluster = list()
            for distanceIndex, distanceRow in distanceDFCluster.iterrows():
                dataRow = dataFrame.iloc[distanceIndex]
                dataFrameCluster.append(dataRow)
            dataFrameCluster = pandas.DataFrame(dataFrameCluster)
            # print(dataFrameCluster)
            nRows, nCols = dataFrameCluster.shape
            centroid = list()
            for col in range(nCols):
                sumOfValues = dataFrameCluster[col].sum()
                newValue = sumOfValues / nRows
                centroid.append(newValue)
            centroids.append(centroid)
        centroidsDF = pandas.DataFrame(centroids)
        # print(centroids)

        # get SSE
        sse = getSSE(distanceDF)

        iter_count = iter_count + 1

    '''
    print("\nFinal centroids:\n", centroids)
    print("\nCentroid cluster:\n", distanceDF)
    print("\nNumber of iterations: ", iter_count)
    print("\nSSE:", sse)
    '''
    return sse


for kValue in [3, 5, 7]:
    sseList = list()
    for iterNum in range(num_iter_k):
        sseList.append(kmeans(kValue))
    averageSSE = mean(sseList)
    print("\nAverage SSE for k = ", kValue, " is ", averageSSE)
