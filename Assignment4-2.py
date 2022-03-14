#%% 

import mnist_reader
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# training data
X_train, y_train = mnist_reader.load_mnist('data', kind = 'train')
# training data length (60,000)
dataLength = len(X_train)

# testing data
X_test, y_test = mnist_reader.load_mnist('data', kind = 't10k')
# testing data length (10,000)
testLength = len(X_test)

# amount of features (784)
featureCount = len(X_train[0])

X_train, y_train, X_test, y_test = map(tensor, (X_train, y_train, X_test, y_test))
X_train = X_train.float()
X_test = X_test.float()

# normalization
def normalize(X, mean, std):
    return (X - mean) / std

X_train = normalize(X_train, X_train.mean(), X_train.std())
X_test = normalize(X_test, X_train.mean(), X_train.std())

mpl.rcParams['image.cmap'] = 'gray'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 4 CNN layers
        self.conv1 = nn.Conv2d(1, 8, 5, 2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 2, 1)
        
        # 1 average pooling layer
        # self.pooling = nn.AdaptiveAvgPool2d(1)
        
        # 1 fully connected layer
        self.fc = nn.Linear(32*1*1, 10)
        
    def Forward(self, X):
        # ReLU opertation
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        
        # average pooling
        X = F.avg_pool2d(X, 1)
        
        X = X.view(-1, self.FlatFeatures(X))
        
        return X
        
    def FlatFeatures(self, X):
        size = X.size()[1:]
        featureNum = 1
        for s in size:
            featureNum *= s
        
        return featureNum
    
if __name__ == "__main__":
    model = CNN()
    
    learningRate = 0.05
    epochs = 10
    lossFunction = F.cross_entropy
    optimal = optim.SGD(model.parameters(), lr = learningRate)
    
    testingAccuracyValues = []
    
    for epoch in range(epochs):
        
        model.train()
        
        for i in range((dataLength - 1) // 32 + 1):
            iStart = i * 32
            iEnd = iStart + 32
            
            filterX = X_train[iStart:iEnd].reshape(32, 1, 28, 28)
            filterY = y_train[iStart:iEnd]
            
            loss = lossFunction(model.Forward(filterX), filterY)
            loss.backward()
            optimal.step()
            optimal.zero_grad()
            
        model.eval()
        
        with torch.no_grad():
            totalLoss = 0.0
            testingAccuracy = 0.0
            
            for i in range(testLength):
                X = X_test[i].reshape(1, 1, 28, 28)
                Y = y_test[i]
                predict = model.Forward(X)
                testingAccuracy += (torch.argmax(predict) == Y).float()
            
            testingAccuracy = (testingAccuracy * 100 / testLength).item()
            print("Testing Accuracy =", testingAccuracy)
            testingAccuracyValues.append(testingAccuracy)
            
    plt.plot([i for i in range(1, 11)], testingAccuracyValues)
    plt.title("Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("Testing Accuracy")
    plt.show()

# %%
