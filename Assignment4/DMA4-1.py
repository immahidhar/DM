from numpy import mean
from sklearn import svm

from loader import loadProblem1DataSets

# constants
numberOfClasses = 6
polynomialKernel = "poly"
polynomialKernelParam = 2
gaussianKernel = "rbf"
gaussianKernelParam = 2

# load problem 1 dataset
x_test, x_train, y_test, y_train = loadProblem1DataSets()


# Trains SVM based on given kernel and degree
def trainSVM(kernel, param):
    x_test_predictions = list()

    # Start loop and model svm for each class separately
    for cls in range(numberOfClasses):
        # learn svm
        if kernel == polynomialKernel:
            svmClassifier = svm.SVC(kernel=kernel, degree=param)
        elif kernel == gaussianKernel:
            svmClassifier = svm.SVC(kernel=kernel, C=param)
        else:
            print("Invalid kernel given")
            return
        x_train_labels = [y[cls] for y in y_train]
        svmClassifier.fit(x_train, x_train_labels)
        # predict from svm
        x_test_prediction = svmClassifier.predict(x_test)
        x_test_predictions.append(x_test_prediction)

    # get predictions for all classes as an array and compute accuracies for each sample
    accuracies = list()
    for sampleIndex in range(len(x_test)):
        x_test_prediction_all_classes = list()
        for cls in range(numberOfClasses):
            x_test_prediction_all_classes.append(x_test_predictions[cls][sampleIndex])
        y_test_all_classes = y_test[sampleIndex]
        tip = 0
        tup = 0
        for cls in range(numberOfClasses):
            if x_test_prediction_all_classes[cls] == 1 and x_test_prediction_all_classes[cls] == y_test_all_classes[cls]:
                tip = tip + 1
            if x_test_prediction_all_classes[cls] == 1 or y_test_all_classes[cls] == 1:
                tup = tup + 1
        accuracy = tip / tup
        accuracies.append(accuracy)

    print("Accuracies of samples = ", accuracies)
    averageAccuracy = mean(accuracies)
    print("Average accuracy % = ", averageAccuracy*100, "%")


print("\nTraining SVM with polynomial kernel with parameter ", polynomialKernelParam)
trainSVM(polynomialKernel, polynomialKernelParam)

print("\nTraining SVM with gaussian kernel with parameter ", gaussianKernelParam)
trainSVM(gaussianKernel, gaussianKernelParam)
