
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (699, 11)
originData = pd.read_csv('data/breast-cancer-wisconsin.data', names= ['Samplecodenumber', 'ClumpThickness', 'UniformityofCellSize', 'UniformityofCellShape', 'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses', 'Class'], header=None)
data = originData.drop('Samplecodenumber', axis=1)
data = data.rename(columns={'Class': 'label'})
data.BareNuclei= data.BareNuclei.replace('?', 1)
data.BareNuclei = pd.to_numeric(data.BareNuclei)

# 699
dataLength = len(data)

# first 70%
X_train = data[ :490]
# next 10%
X_valid = data[490:560]
# last 20%
X_test = data[560: ]

class DecisionTree:

    def Entropy(self, data):
        label = data[:-1]
        labels, labelCount = np.unique(label, return_counts=True)
        p = labelCount/labelCount.sum()
        entropy = sum(p * -np.log2(p))

        return entropy

    def InformationGain(self, leftSub, rightSub):
        leftProb = len(leftSub) / (len(leftSub) + len(rightSub))
        rightProb = len(rightSub) / (len(leftSub) + len(rightSub))

        infoGain = leftProb * self.Entropy(leftSub) + rightProb * self.Entropy(rightSub)

        return infoGain

    def Classify(self, newList, tree):
        q = list(tree.keys())[0]
        attributes, _, value = q.split()

        # go left or go right
        if newList[attributes] < float(value):
            predict = tree[q][0]
        else:
            predict = tree[q][1]

        # reach leaf node
        if not isinstance(predict, dict):
            return predict
        # still have child
        else:
            subTree = predict
        
        return self.Classify(newList, subTree)


    def Predict(self, data, tree):
        data['prediction'] = data.apply(self.Classify, axis=1, args=(tree,))
        data['score'] = data.prediction == data.label
        score = data.score.mean()
        
        return score

    def FindLabel(self, data):
        label = data[:,-1]
        labels, labelCount = np.unique(label, return_counts=True)
        labelFound = labels[labelCount.argmax()]

        return labelFound

    def ID3(self, data, level = 0, samples = 0, depth = 0):
        global titles
        if level == 0:
            titles = data.columns
            data = data.to_numpy()
        else:
            data = data

        # get all the labels
        label = data[:,-1]
        labelValues = np.unique(label)

        # the child are all in the same class
        if len(labelValues) == 1:
            isComplete = True
        else:
            isComplete = False

        if level == depth or len(data) < samples or isComplete == True:
            classifiedLabel = self.FindLabel(data)
            return classifiedLabel

        else:
            level += 1
            split = {}
            xLength, yLength = data.shape

            for column in range(yLength-1):
                split[column] = []
                attributeValue = data[:, column]
                valueList = np.unique(attributeValue)

                for f in range(len(valueList)):
                    if f != 0:
                        fi = valueList[f]
                        fj = valueList[f-1]
                        # the candidate threshold
                        threshold = int(fj) + (int(fi) - int(fj))/2
                        split[column].append(threshold)

            entropy = 100

            for i in split:
                for j in split[i]:
                    columnData = data[:, i]

                    leftTree = data[columnData <= j]
                    rightTree = data[columnData > j]

                    newEntropy = self.InformationGain(leftTree, rightTree)

                    # more information gain
                    if newEntropy < entropy:
                        entropy = newEntropy
                        target = i
                        targetValue = j

            dataLabel = data[:, target]
            leftLabel = data[dataLabel <= targetValue]
            rightLabel = data[dataLabel > targetValue]

            targetAttribute = titles[target]
            q = "{} <= {}".format(targetAttribute, targetValue)
            tree = {q: []}

            leftBranch = self.ID3(leftLabel, level, samples, depth)
            rightBranch = self.ID3(rightLabel, level, samples, depth)
            tree[q].append(leftBranch)
            tree[q].append(rightBranch)

        return tree

if __name__ == "__main__":
    clf = DecisionTree()

    print("start")
    tree = clf.ID3(X_train, samples=3, depth=10)
    print("tree built")

    print("predicting...")
    validationAccuracy = clf.Predict(X_valid, tree)
    testAccuracy = clf.Predict(X_test, tree)

    print("Validation Accuracy =", validationAccuracy)
    print("Testing Accuracy =", testAccuracy)

    validAccuracy = []
    testAccuracy = []

    # pruning
    for maxDepth in range(2, 15):
        tree = clf.ID3(X_train, samples=3, depth=maxDepth)
        
        validAccuracy.append(clf.Predict(X_valid, tree))
        testAccuracy.append(clf.Predict(X_test, tree))

    plt.figure()
    plt.plot(np.arange(2, 15, 1), validAccuracy, 'b', label="Validation")
    plt.plot(np.arange(2, 15, 1), testAccuracy, 'r', label="Testing")
    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("3_Pruning Accuracy")


        

# %%
