# %%

import mnist_reader
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

import time

# training data
X, y = mnist_reader.load_mnist('data', kind='train')
# training data length (60,000)
dataLength = len(X)
# amount of features (784)
featureNum = len(X[0])

trainDataL = int(0.8 * dataLength)

# first 80% for training
X_train = X[:trainDataL]
y_train = y[:trainDataL]

# last 20% for validation
X_valid = X[trainDataL:]
y_valid = y[trainDataL:]

# testing data
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')
# testing data length (10,000)
testLength = len(X_test)


def KernelizedPerceptron():
    #                                                  48,000 , 784
    X_train_normalized = X_train / (np.zeros(shape=(trainDataL, featureNum)) + 255)
    alpha = np.zeros(shape=(trainDataL, 10))

    trainMistakes = []

    # degree of 2, 3, and 4
    for degree in range(2, 5):
        print(time.asctime(time.localtime(time.time())))
        print("Degree =", degree, "start")

        # 5 iterations
        for i in range(5):
            print(time.asctime(time.localtime(time.time())))
            print("i =", i, "start")

            mistake = 0

            for j in range(trainDataL):

                # predict
                yHat = (np.dot(X_train_normalized, X_train_normalized[j])+1)**degree
                yHat = np.dot(yHat, alpha).argmax()

                # mistake
                if yHat != y_train[j]:
                    mistake += 1
                    alpha[j, y_train[j]] += 1
                    alpha[j, yHat] -= 1

            trainMistakes.append(mistake)

            print(time.asctime(time.localtime(time.time())))
            print("i =", i, "end")

        weight = np.dot(alpha.T, X_train)

        # training accuracy
        for predict in np.dot(X_train, weight.T):
            trainPredict = [predict.argmax()]

        trainAccuracy = np.sum(trainPredict == y_train) / trainDataL

        # validation accuracy
        for predict in np.dot(X_valid, weight.T):
            validPredict = [predict.argmax()]

        validAccuracy = np.sum(validPredict == y_valid) / (dataLength - trainDataL)

        # testing accuracy
        for predict in np.dot(X_test, weight.T):
            testPredict = [predict.argmax()]

        testAccuracy = np.sum(testPredict == y_test) / testLength

        print("Degree", degree, "Training accuracy =", trainAccuracy)
        print("Degree", degree, "Validation accuracy =", validAccuracy)
        print("Degree", degree, "Testing accuracy =", testAccuracy)

        print(time.asctime(time.localtime(time.time())))
        print("Degree =", degree, "end")

    print(trainMistakes)

    plt.figure()
    plt.plot(np.arange(1, 6, 1), trainMistakes[0:5], 'b', label="Degree = 2")
    plt.plot(np.arange(1, 6, 1), trainMistakes[5:10], 'r', label="Degree = 3")
    plt.plot(np.arange(1, 6, 1), trainMistakes[10:15], 'g', label="Degree = 4")
    plt.xlabel("Iterations")
    plt.ylabel("Mistakes")
    plt.legend()
    plt.savefig("2_Training_mistakes")


if __name__ == "__main__":

    KernelizedPerceptron()


# %%
