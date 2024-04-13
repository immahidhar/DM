import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from loader import loadProblem3DataSet

# constraints
positiveSubSetPercent = 10
trainingPercent = 70
overSamplingEnabled = True

# load credit card dataset
dataFrame = loadProblem3DataSet()

# separate training and testing data
print("Separating training and testing data ...")
x = dataFrame.drop('Class', axis=1).values
y = dataFrame[['Class']].values.reshape(-1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=((100 - trainingPercent) / 100), stratify=y)

# oversampling positive class sample data
if overSamplingEnabled:
    print("Performing oversampling on training data ...")
    x_train_DF = pandas.DataFrame(x_train)
    x_train_DF['Class'] = y_train
    x_pSampleDF = x_train_DF[x_train_DF['Class'] == 1]
    pSampleCount = x_pSampleDF.shape[0]
    nSampleCount = x_train_DF.shape[0] - pSampleCount
    os_pSampleCount = round((positiveSubSetPercent * nSampleCount) / (100 - positiveSubSetPercent))
    print("Positive sample count = ", pSampleCount)
    print("Negative sample count = ", nSampleCount)
    print("Positive sample count after oversampling = ", os_pSampleCount)
    while x_pSampleDF.shape[0] < os_pSampleCount:
        x_pSampleDF = pandas.concat([x_pSampleDF, x_pSampleDF])
    x_pSampleDF.reset_index(drop=True, inplace=True)
    if x_pSampleDF.shape[0] > os_pSampleCount:
        x_pSampleDF.drop(range(os_pSampleCount - pSampleCount - 1, x_pSampleDF.shape[0] - 1), axis=0, inplace=True)
    x_train_DF = pandas.concat([x_train_DF, x_pSampleDF])
    x_train_DF.reset_index(drop=True, inplace=True)
    print(x_train_DF, "\n")
    x_train = x_train_DF.drop('Class', axis=1).values
    y_train = x_train_DF[['Class']].values.reshape(-1)

# Random forest classifier
print("Training a Random Forest Classifier ...")
rfc = RandomForestClassifier()
# fit the predictor and target
rfc.fit(x_train, y_train)

# predict
print("Predicting labels of test data ...")
rfc_predict = rfc.predict(x_test)

# check performance
print("Checking performance ...")
print('ROC AUC score:', roc_auc_score(y_test, rfc_predict))
print('Accuracy score:', accuracy_score(y_test, rfc_predict))
print('F score:', f1_score(y_test, rfc_predict))
