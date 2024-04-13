import pandas
from scipy import io

# TODO : replace below with the path where the data set folders are present
dataPath = "/Users/mahidharreddynarala/Desktop/Stuff/FSU/Fall'22 sem/Data Mining/Assignments/Assignment 4/Data for " \
           "Assignment 4/"
problem1Path = "Data for Problem 1/"
problem2Path = "Data for Problem 2/"
problem3Path = "Data for Problem 3/"


# load problem 1 data sets
def loadProblem1DataSets():
    print("Loading Problem 1 datasets ...")
    x_test = list()
    x_train = list()
    y_test = list()
    y_train = list()
    x_test.append((io.loadmat(dataPath + problem1Path + "X_test.mat"))["X_test"])
    x_train.append((io.loadmat(dataPath + problem1Path + "X_train.mat"))["X_train"])
    y_test.append((io.loadmat(dataPath + problem1Path + "Y_test.mat"))["y_test"])
    y_train.append((io.loadmat(dataPath + problem1Path + "Y_train.mat"))["y_train"])
    print("Problem 1 datasets loaded")
    return x_test[0], x_train[0], y_test[0], y_train[0]


# load problem 2 data set
def loadProblem2DataSet():
    print("Loading Problem 2 dataset ...")
    file = open(dataPath + problem2Path + "seeds.txt", "r")
    dataSet = list()
    for data in file:
        if data is not None and data != "" and data != "\n":
            data = data.split()
            data = [float(value) for value in data]
            dataSet.append(data)
    print("Problem 2 dataset loaded")
    return pandas.DataFrame(dataSet)


# load problem 3 data set
def loadProblem3DataSet():
    print("Loading Problem 3 dataset ...")
    df = pandas.read_csv(dataPath + problem3Path + "creditcard.csv")
    print("Problem 3 dataset loaded")
    return df
