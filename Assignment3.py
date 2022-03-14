#%%

import numpy as np
import collections

# training data
trainX = np.genfromtxt("fortune-cookie-data/traindata.txt", dtype="str", delimiter="\t")
# 322
trainLength = len(trainX)
# training label
trainY = np.genfromtxt("fortune-cookie-data/trainlabels.txt", dtype="int", delimiter="\t")
# testing data
testX = np.genfromtxt("fortune-cookie-data/testdata.txt", dtype="str", delimiter="\t")
#101
testLength = len(testX)
# testing label
testY = np.genfromtxt("fortune-cookie-data/testlabels.txt", dtype="int", delimiter="\t")
# stop list
stopList = np.genfromtxt("fortune-cookie-data/stoplist.txt", dtype="str", delimiter="\t")

# create a dictionary with vocabularies in the stop list as keys with a value of 1
stopDict = collections.Counter()
for i in stopList:
    stopDict[i] = 1

def MakeDict(X):
    dictX = collections.Counter()
    for i in X:
        # turn messages into keys
        keys = RemoveStopWord(i)
        for j in keys:
            dictX[j] += 1

    return dictX

# split the messages in to vocabularies and remove the stop words
def RemoveStopWord(message):
    message = message.lower()
    vocabs = message.split()

    i = len(vocabs)
    j = 0

    while j < i:
        if vocabs[j] in stopDict:
            del vocabs[j]
            i -= 1
            j -= 1
        j += 1

    return vocabs

# calculate probability
def PCount(message, vocabs, vocabLen, len):

    # remove stop word and split message
    splitMessage = RemoveStopWord(message)
    for i in splitMessage:
        # count words that are not in the dictionary
        if i not in vocabs:
            vocabLen += 1

    result = 1
    # count percentage
    for i in splitMessage:
        if i in vocabs:
            result = result * vocabs[i] / vocabLen
        else:
            result = result / vocabLen

    return result * len 

if __name__ == "__main__":
    # the first 152 training data are labeled as 1
    trainXOne = trainX[:152]
    # the rest of the training data (170) are labeled as 0
    trainXZero = trainX[152:]

    vocabCountOne = MakeDict(trainXOne)
    vocabCountZero = MakeDict(trainXZero)
    vocabCountTrain = MakeDict(trainX)

    # calculate training accuracy
    counter = 0
    for i in range(trainLength):
        pOne = PCount(trainX[i], vocabCountOne, len(vocabCountTrain), 152/trainLength)
        pZero = PCount(trainX[i], vocabCountZero, len(vocabCountTrain), 170/trainLength)

        yHatOne = pOne / (pOne + pZero)
        yHatZero = pZero / (pOne + pZero)

        # predict label 1
        if yHatOne > yHatZero and trainY[i] == 1:
            counter += 1
        # predict label 0
        elif yHatZero > yHatOne and trainY[i] == 0:
            counter += 1

    trainAccuracy = counter / trainLength
    print("Training Accuracy :", trainAccuracy)

    # calculate testing accuracy
    counter = 0
    for i in range(testLength):
        pOne = PCount(testX[i], vocabCountOne, len(vocabCountTrain), 152/trainLength)
        pZero = PCount(testX[i], vocabCountZero, len(vocabCountTrain), 170/trainLength)

        yHatOne = pOne / (pOne + pZero)
        yHatZero = pZero / (pOne + pZero)

        # predict label 1
        if yHatOne > yHatZero and testY[i] == 1:
            counter += 1
        # predict label 0
        elif yHatZero > yHatOne and testY[i] == 0:
            counter += 1

    testAccuracy = counter / testLength
    print("Testing Accuracy :", testAccuracy)

    print("output.txt created")

    file = open("output.txt", "w+")
    file.write("Training Accuracy : " + str(trainAccuracy) + "\n")
    file.write("Testing Accuracy : " + str(testAccuracy))
    file.close()

# %%
