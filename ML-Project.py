"""
Author: Kajal Gupta
Date: 11/23/2019
"""
import random
from math import inf

from sklearn import model_selection
from sklearn.svm import LinearSVC
import sys

data = open(sys.argv[1])
labels = open(sys.argv[2])
testData = open(sys.argv[3])

dataMatrix = []
for line in data.readlines():
    line = [float(i) for i in line.split()]
    dataMatrix.append(line)

labelsInfo = {}
for line in labels.readlines():
    val, key = line.split()
    labelsInfo.update({int(key): int(val)})

testDataMatrix = []
for line in testData.readlines():
    line = [int(i) for i in line.split()]
    testDataMatrix.append(line)

print("Data Loaded: train, labels, test ", len(labelsInfo), len(testDataMatrix), len(dataMatrix))


def getbest(train, labels):
    c = [.001, .01, .1, 1, 10, 100]
    error = {}
    for i in c:
        error[i] = 0

    for i in c:
        model = LinearSVC(max_iter=100000, C=i)
        scoreNew = model_selection.cross_val_score(model, train, list(labels.values()), cv=10)
        error[i] = 1 - max(scoreNew)

    minError = float(inf)
    bestC = 0
    for key in error.keys():
        if error[key] < minError:
            minError = error[key]
            bestC = key

    return bestC, minError*100


def trainModel(trainData, labels):
    """
    Accepts train dataset and its label files for initiating training.
    :return: trained weights
    """
    c, _ = getbest(trainData, labels)
    print(c)
    svm = LinearSVC(max_iter=100000, C=c)
    model = svm.fit(trainData, list(labels.values()))
    return model


def testModel(model, testData):
    """
    Performs testing using the trained model
    :param model: trained weights
    :param testData: test data to perform testing on
    :return: returns predictions of test dataset. Its a python dictionary
    """
    predictions = model.predict(testData)
    return predictions


def calculateAccuracy(testPredictions, actualPredictions):
    """
    Checks accuracy of the trained models
    :param testPredictions: output predictions
    :param actualPredictions: actual predictions
    :return: accuracy, precision, recall
    """
    confusionMatrix = [[0, 0], [0, 0]]
    for i in range(len(testPredictions)):
        if actualPredictions[i] == 0 and testPredictions[i] == 0:
            confusionMatrix[0][0] += 1
        elif actualPredictions[i] == 1 and testPredictions[i] == 1:
            confusionMatrix[1][1] += 1
        elif actualPredictions[i] == 0 and testPredictions[i] == 1:
            confusionMatrix[0][1] += 1
        elif actualPredictions[i] == 1 and testPredictions[i] == 0:
            confusionMatrix[1][0] += 1

    print(confusionMatrix)
    accuracy = (confusionMatrix[0][0] + confusionMatrix[1][1])/(sum(confusionMatrix[0]) + sum(confusionMatrix[1]))
    error = 1 - accuracy
    return accuracy*100, error*100


def calculateCorrelation(colI, colJ):
    """
    Calculates correlation for two columns.
    :param colI: first col
    :param colJ: second col
    :return: returns correlation cofficient
    """
    uniqColI = set(colI)
    uniqColJ = set(colJ)
    contigencyMatrix = [[0 for i in range((len(uniqColJ)+1))] for j in range((len(uniqColI)+1))]

    for i, valI in enumerate(uniqColI):
        for j, valJ in enumerate(uniqColJ):
            for colVal in range(0, len(colI)):
                if colI[colVal] == valI and colJ[colVal] == valJ:
                    contigencyMatrix[i][j] += 1
                    contigencyMatrix[len(uniqColI)][j] += 1
        contigencyMatrix[i][len(uniqColJ)] = sum(contigencyMatrix[i])

    contigencyMatrix[len(uniqColI)][len(uniqColJ)] = sum(contigencyMatrix[len(uniqColI)])

    observedMatrxix = [[0 for i in range(len(uniqColJ))] for j in range(len(uniqColI))]

    for i, valI in enumerate(uniqColI):
        for j, valJ in enumerate(uniqColJ):
            observedMatrxix[i][j] = (contigencyMatrix[i][len(uniqColJ)]*contigencyMatrix[len(uniqColI)][j])/contigencyMatrix[len(uniqColI)][len(uniqColJ)]

    chiSq = 0
    for i in range(0, len(uniqColI)):
        for j in range(0, len(uniqColJ)):
            try:
                chiSq += (contigencyMatrix[i][j] - observedMatrxix[i][j])**2/observedMatrxix[i][j]
            except Exception as e:
                pass

    significanceFromTable = 18.465  # at significance of 0.001 and dof 4
    if chiSq > significanceFromTable:
        return 1, chiSq
    else:
        return -1, chiSq


def performFeatureSelection(dataMatrix, testDataMatrix, trainingLabels):
    print("Performing feature selection, it might take some time!")
    columns = len(dataMatrix[0])
    chiValues = []
    for col in range(0, columns):
        colI = [dataMatrix[row][col] for row in range(0, len(dataMatrix))]
        correlation, chiSq = calculateCorrelation(colI, list(trainingLabels.values()))
        chiValues.append((col, correlation, chiSq))

    features = 25
    print("No. of features selected = ", features)
    chiValues = sorted(chiValues, reverse=True, key=lambda x: x[2])[:features]
    columnsToPick = [j[0] for j in chiValues]

    dataMatrixNew = [[] for i in range(0, len(dataMatrix))]
    for i, row in enumerate(dataMatrix):
        for j, col in enumerate(row):
            if j in columnsToPick:
                dataMatrixNew[i].append(col)

    testDataMatrixNew = [[] for i in range(0, len(testDataMatrix))]
    for i, row in enumerate(testDataMatrix):
        for j, col in enumerate(row):
            if j in columnsToPick:
                testDataMatrixNew[i].append(col)

    return dataMatrixNew, testDataMatrixNew


def startProject(dataMatrix, labelsInfo, testDataMatrix):
    """
    Actual function that takes whole dataset,
    does feature selection and then trains the model on reduced features,
    performs testing on testData and returns final predictions
    :param dataMatrix: train dataset
    :param labelsInfo: data labels
    :param testDataMatrix: test dataset
    :return: predictions of test dataset
    """
    trainData, testData = performFeatureSelection(dataMatrix, testDataMatrix, labelsInfo)
    model = trainModel(trainData, labelsInfo)
    predictions = testModel(model, testData)
    return predictions


def printPredictions(predictions, noLabels=True):
    """
    Outputs predictions on console
    :param predictions: predictions dictionary
    :return: None
    """
    if noLabels:
        for i, val in enumerate(predictions):
            print(val, i)
    else:
        for key in predictions:
            print(predictions[key], key)


def testTraining(dataMatrix, labelsInfo, k):
    """
    This function is used to perform validations to be assured whether the algorithm is
    working correctly or not. This is test function used by me to validate my model and algorithm
    and should not be used by TAs. The actual function for project output is startProject
    :param dataMatrix: training data
    :param labelsInfo: training labels
    :param k: training:testing ratio
    :return: None
    """
    # splitting the data in test and train and then starting feature selection
    dataIndexes = [i for i in range(0, len(dataMatrix)) if labelsInfo.get(i) != None]
    trainingDataIndexes = random.sample(dataIndexes, int(len(dataMatrix) * k / 100))

    trainingData, trainingLabels, trainI = [], {}, 0
    testData, testLabels, testI = [], {}, 0
    for i, row in enumerate(dataMatrix):
        if labelsInfo.get(i) != None:
            if i in trainingDataIndexes:
                trainingData.append(row)
                trainingLabels.update({trainI: labelsInfo.get(i)})
                trainI += 1
            else:
                testData.append(row)
                testLabels.update({testI: labelsInfo.get(i)})
                testI += 1

    print(len(trainingLabels), len(trainingData), trainI)
    print(len(testLabels), len(testData), testI)
    testPredictions = startProject(trainingData, trainingLabels, testData)
    accuracy, error = calculateAccuracy(testPredictions, list(testLabels.values()))
    print("accuracy= {}, error= {}".format(accuracy, error))
    # printPredictions(testPredictions, False)
    return accuracy, error


if __name__ == '__main__':
    """
    Use testTraining to do in house training and validation while project development
    Use startProject to do predictions on actual test set by TAs
    """

    #cross validation for my testing and training
    acc, err = [], []
    for i in range(0, 5):
        _acc, _err = testTraining(dataMatrix, labelsInfo, 70)
        acc.append(_acc)
        err.append(_err)

    print("avg accuracy = {}, avg error = {}".format(sum(acc)/5, sum(err)/5))

    predictions = startProject(dataMatrix, labelsInfo, testDataMatrix)
    for i, val in enumerate(predictions):
        print(val, i)

