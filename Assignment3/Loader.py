from scipy import io

# TODO : replace below with the path where the data set folders are present
dataPath = "/Users/mahidharreddynarala/Desktop/Stuff/FSU/Fall'22 sem/Data Mining/Assignments/Assignment 3/Data for " \
           "Assignment 3/"
dataSet1Name = "MMI/"
dataSet2Name = "MindReading/"


# read .mat file
def readMatFile(filePath):
    mat = io.loadmat(filePath)
    return mat


# load mmi data sets
def loadMMIDataSets():
    print("Loading MMI dataset ...")
    mmiTrainingSets = list()
    mmiTrainingLabels = list()
    mmiUnlabeledSets = list()
    mmiUnlabeledLabels = list()
    mmiTestingSets = list()
    mmiTestingLabels = list()

    mmiTrainingSets.append((readMatFile(dataPath + dataSet1Name + "trainingMatrix_1.mat"))["trainingMatrix"])
    mmiTrainingSets.append((readMatFile(dataPath + dataSet1Name + "trainingMatrix_2.mat"))["trainingMatrix"])
    mmiTrainingSets.append((readMatFile(dataPath + dataSet1Name + "trainingMatrix_3.mat"))["trainingMatrix"])
    mmiTrainingLabels.append((readMatFile(dataPath + dataSet1Name + "trainingLabels_1.mat"))["trainingLabels"])
    mmiTrainingLabels.append((readMatFile(dataPath + dataSet1Name + "trainingLabels_2.mat"))["trainingLabels"])
    mmiTrainingLabels.append((readMatFile(dataPath + dataSet1Name + "trainingLabels_3.mat"))["trainingLabels"])
    mmiUnlabeledSets.append((readMatFile(dataPath + dataSet1Name + "unlabeledMatrix_1.mat"))["unlabeledMatrix"])
    mmiUnlabeledSets.append((readMatFile(dataPath + dataSet1Name + "unlabeledMatrix_2.mat"))["unlabeledMatrix"])
    mmiUnlabeledSets.append((readMatFile(dataPath + dataSet1Name + "unlabeledMatrix_3.mat"))["unlabeledMatrix"])
    mmiUnlabeledLabels.append((readMatFile(dataPath + dataSet1Name + "unlabeledLabels_1.mat"))["unlabeledLabels"])
    mmiUnlabeledLabels.append((readMatFile(dataPath + dataSet1Name + "unlabeledLabels_2.mat"))["unlabeledLabels"])
    mmiUnlabeledLabels.append((readMatFile(dataPath + dataSet1Name + "unlabeledLabels_3.mat"))["unlabeledLabels"])
    mmiTestingSets.append((readMatFile(dataPath + dataSet1Name + "testingMatrix_1.mat"))["testingMatrix"])
    mmiTestingSets.append((readMatFile(dataPath + dataSet1Name + "testingMatrix_2.mat"))["testingMatrix"])
    mmiTestingSets.append((readMatFile(dataPath + dataSet1Name + "testingMatrix_3.mat"))["testingMatrix"])
    mmiTestingLabels.append((readMatFile(dataPath + dataSet1Name + "testingLabels_1.mat"))["testingLabels"])
    mmiTestingLabels.append((readMatFile(dataPath + dataSet1Name + "testingLabels_2.mat"))["testingLabels"])
    mmiTestingLabels.append((readMatFile(dataPath + dataSet1Name + "testingLabels_3.mat"))["testingLabels"])

    return mmiTrainingSets, mmiTrainingLabels, mmiUnlabeledSets, mmiUnlabeledLabels, mmiTestingSets, mmiTestingLabels


# load mind reading data sets
def loadMindReadingDataSets():
    print("Loading Mind Reading dataset ...")
    mrTrainingSets = list()
    mrTrainingLabels = list()
    mrUnlabeledSets = list()
    mrUnlabeledLabels = list()
    mrTestingSets = list()
    mrTestingLabels = list()

    mrTrainingSets.append((readMatFile(dataPath + dataSet2Name + "trainingMatrix_MindReading1.mat"))["trainingMatrix"])
    mrTrainingSets.append((readMatFile(dataPath + dataSet2Name + "trainingMatrix_MindReading2.mat"))["trainingMatrix"])
    mrTrainingSets.append((readMatFile(dataPath + dataSet2Name + "trainingMatrix_MindReading3.mat"))["trainingMatrix"])
    mrTrainingLabels.append((readMatFile(dataPath + dataSet2Name + "trainingLabels_MindReading_1.mat"))["trainingLabels"])
    mrTrainingLabels.append((readMatFile(dataPath + dataSet2Name + "trainingLabels_MindReading_2.mat"))["trainingLabels"])
    mrTrainingLabels.append((readMatFile(dataPath + dataSet2Name + "trainingLabels_MindReading_3.mat"))["trainingLabels"])
    mrUnlabeledSets.append((readMatFile(dataPath + dataSet2Name + "unlabeledMatrix_MindReading1.mat"))["unlabeledMatrix"])
    mrUnlabeledSets.append((readMatFile(dataPath + dataSet2Name + "unlabeledMatrix_MindReading2.mat"))["unlabeledMatrix"])
    mrUnlabeledSets.append((readMatFile(dataPath + dataSet2Name + "unlabeledMatrix_MindReading3.mat"))["unlabeledMatrix"])
    mrUnlabeledLabels.append((readMatFile(dataPath + dataSet2Name + "unlabeledLabels_MindReading_1.mat"))["unlabeledLabels"])
    mrUnlabeledLabels.append((readMatFile(dataPath + dataSet2Name + "unlabeledLabels_MindReading_2.mat"))["unlabeledLabels"])
    mrUnlabeledLabels.append((readMatFile(dataPath + dataSet2Name + "unlabeledLabels_MindReading_3.mat"))["unlabeledLabels"])
    mrTestingSets.append((readMatFile(dataPath + dataSet2Name + "testingMatrix_MindReading1.mat"))["testingMatrix"])
    mrTestingSets.append((readMatFile(dataPath + dataSet2Name + "testingMatrix_MindReading2.mat"))["testingMatrix"])
    mrTestingSets.append((readMatFile(dataPath + dataSet2Name + "testingMatrix_MindReading3.mat"))["testingMatrix"])
    mrTestingLabels.append((readMatFile(dataPath + dataSet2Name + "testingLabels_MindReading1.mat"))["testingLabels"])
    mrTestingLabels.append((readMatFile(dataPath + dataSet2Name + "testingLabels_MindReading2.mat"))["testingLabels"])
    mrTestingLabels.append((readMatFile(dataPath + dataSet2Name + "testingLabels_MindReading3.mat"))["testingLabels"])

    return mrTrainingSets, mrTrainingLabels, mrUnlabeledSets, mrUnlabeledLabels, mrTestingSets, mrTestingLabels
