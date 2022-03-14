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


def LinearSVM():

    trainAccuracy = []
    validAccuracy = []
    testAccuracy = []

    iteration = []

    supportVectorNum = []

    maxValidAcc = 0

    for i in np.arange(-4, 5, dtype=float):

        # 10^4 to 10^4
        C = 10**i
        iteration.append(str(C))

        # train classifier
        clf = SVC(kernel='linear', C=C, max_iter=10000)
        clf.fit(X_train, y_train)

        # accuracy for training, validation, and testing
        trainAccuracy.append(clf.score(X_train, y_train))
        validAccuracy.append(clf.score(X_valid, y_valid))
        testAccuracy.append(clf.score(X_test, y_test))

        # number of support vectors
        supportVectorNum.append(clf.n_support_)

        # get the best C
        if clf.score(X_valid, y_valid) > maxValidAcc:
            maxValidAcc = clf.score(X_valid, y_valid)
            bestC = C

        print(time.asctime(time.localtime(time.time())))
        print("i =", int(i))

    plt.figure()
    plt.plot(iteration, trainAccuracy, 'r', label="Training accuracy")
    plt.plot(iteration, validAccuracy, 'g', label="Validation accuracy")
    plt.plot(iteration, testAccuracy, 'b', label="Testing accuracy")
    plt.ylim([0, 1])
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("1a_LinearSVM_limit")

    plt.figure()
    plt.plot(iteration, supportVectorNum, 'b',
             label="Number of Support Vectors")
    plt.xlabel("C")
    plt.ylabel("Amount")
    plt.savefig("1a_LinearSVM_SVcount_limit")

    return bestC


def ConfusionMatrix(bestC):

    clf = SVC(kernel='linear', C=bestC, max_iter=10000)
    clf.fit(X, y)

    # testing accuracy
    accuracy = clf.score(X_test, y_test)

    print(time.asctime(time.localtime(time.time())))
    print("C =", bestC)
    print("Testing accuracy:", accuracy)

    plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues, normalize='all')
    plt.title("Confusion Matrix")
    plt.savefig("1b_ConfusionMatrix_limit")


def PolynomialKernel(bestC):

    trainAccuracy = []
    validAccuracy = []
    testAccuracy = []

    supportVectorNum = []

    # Linear SVM
    clf = SVC(kernel='linear', C=bestC, max_iter=10000)
    clf.fit(X_train, y_train)

    # accuracy for training, validation, and testing
    trainAccuracy.append(clf.score(X_train, y_train))
    validAccuracy.append(clf.score(X_valid, y_valid))
    testAccuracy.append(clf.score(X_test, y_test))

    # number of support vectors
    supportVectorNum.append(clf.n_support_)

    # kernel of degree 2, 3, and 4
    for degree in range(2, 5):

        clf = SVC(kernel='poly', degree=degree, C=bestC, max_iter=10000)
        clf.fit(X_train, y_train)

        # accuracy for training, validation, and testing
        trainAccuracy.append(clf.score(X_train, y_train))
        validAccuracy.append(clf.score(X_valid, y_valid))
        testAccuracy.append(clf.score(X_test, y_test))

        # number of support vectors
        supportVectorNum.append(clf.n_support_)

        print(time.asctime(time.localtime(time.time())))
        print("Degree =", degree)

    plt.figure()
    plt.plot(np.arange(1, 5), trainAccuracy, 'r', label="Training accuracy")
    plt.plot(np.arange(1, 5), validAccuracy, 'g', label="Validation accuracy")
    plt.plot(np.arange(1, 5), testAccuracy, 'b', label="Testing accuracy")
    plt.ylim([0, 1])
    plt.xlabel("Kernel degree")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("1c_Poly_Kernel_Accuracy_limit")

    plt.figure()
    plt.plot(np.arange(1, 5), supportVectorNum,
             'b', label="Number of Support Vectors")
    plt.xlabel("Kernel degree")
    plt.ylabel("Amount")
    plt.savefig("1c_Poly_Kernel_SVcount_limit")


if __name__ == "__main__":

    print(time.asctime(time.localtime(time.time())))

    # (a)
    print("a")
    bestC = LinearSVM()

    # (b)
    print("b")
    ConfusionMatrix(bestC)

    # (c)
    print("c")
    PolynomialKernel(bestC)


# %%
