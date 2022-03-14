#%%

import mnist_reader
import numpy as np
import matplotlib.pyplot as pyplot

# training data
X_train, y_train = mnist_reader.load_mnist('data', kind = 'train')
# training data length (60,000)
dataLength = len(X_train)
# amount of features (784)
featureNum = len(X_train[0])

# testing data
X_test, y_test = mnist_reader.load_mnist('data', kind = 't10k')
# testing data length (10,000)
testLength = len(X_test)

# labels for even and odd numbers
labelDic = {0: 1, 1: -1, 2: 1, 3: -1, 4: 1, 5: -1, 6: 1, 7: -1, 8: 1, 9: -1}

testAccuracy = [0] * 1000

def testing(weightVector, iteration):
    bingo = 0

    for i in range(testLength):
        # predict
        yt = np.sign(np.dot(weightVector, X_test[i]))

        # correct guess
        if yt == labelDic[y_test[i]]:
            bingo += 1

    testAccuracy[iteration] = bingo / testLength


def BinaryPerceptron(iteration, type, isTesting):

    # for question a. b. and c.
    if type != "d.":
        weightVector = np.array([0] * featureNum)
        mistake = [0] * iteration
        accuracy = [0] * iteration

        for i in range(iteration):
            for j in range(dataLength):

                # prediction
                yt = np.sign(np.dot(weightVector, X_train[j]))

                # mistake and weight update
                if yt != labelDic[y_train[j]]:
                    weightVector = weightVector + (np.dot(labelDic[y_train[j]], X_train[j]))
                    mistake[i] += 1

            # do the testing
            if isTesting:
                testing(weightVector, i)

            accuracy[i] = (dataLength - mistake[i]) / dataLength

    # for question d.
    if type == "d.":
        temp = 600
        d = 100

        for n in range(temp):
            weightVector = np.array([0] * featureNum)

            for i in range(iteration):
                for j in range(d):

                    # prediction
                    yt = np.sign(np.dot(weightVector, X_train[j]))

                    # mistake and weight update
                    if yt != labelDic[y_train[j]]:
                        weightVector = weightVector + (np.dot(labelDic[y_train[j]], X_train[j]))

            testing(weightVector, n)
            d += 100

        #plot
        pyplot.plot([i for i in range(1, temp*100, 100)], testAccuracy[:temp])
        pyplot.title("5.1 d. Binary Perceptron Testing")
        pyplot.xlabel("Training Examples")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('51d Binary Perceptron Testing')
        pyplot.show()
        
    # plot    
    if type == "a.":
        # learning curve, x is iteration, y is amount of mistakes
        pyplot.plot([i for i in range(1, iteration+1)], mistake)
        pyplot.title("5.1 a. Binary Perceptron")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Mistakes")
        pyplot.savefig('51a Binary Perceptron')
        pyplot.show()

    if type == "b.":
        # training accuracy curve, x is iteration, y is accuracy
        pyplot.plot([i for i in range(1, iteration+1)], accuracy)
        pyplot.title("5.1 b. Binary Perceptron Training")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('51b Binary Perceptron Training')
        pyplot.show()

        # testing accuracy curve
        pyplot.plot([i for i in range(1, iteration+1)], testAccuracy[:iteration])
        pyplot.title("5.1 b. Binary Perceptron Testing")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('51b Binary Perceptron Testing')
        pyplot.show()


def PassiveAggressive(iteration, type, isTesting):
    weightVector = np.array([0] * featureNum)
    mistake = [0] * iteration
    accuracy = [0] * iteration

    for i in range(iteration):
        for j in range(dataLength):
            # prediction
            yt = np.sign(np.dot(weightVector, X_train[j]))

            # mistake and weight update
            if yt != labelDic[y_train[j]]:
                tau = (1 - np.dot(labelDic[y_train[j]], np.dot(weightVector, X_train[j]))) / np.linalg.norm(X_train[j]) ** 2
                weightVector = weightVector + np.dot(tau, np.dot(labelDic[y_train[j]], X_train[j]))
                mistake[i] += 1

        # do the testing
        if isTesting:
            testing(weightVector, i)

        accuracy[i] = (dataLength - mistake[i]) / dataLength


    if type == "a.":
        # learning curve, x is iteration, y is amount of mistakes
        pyplot.plot([i for i in range(1, iteration+1)], mistake)
        pyplot.title("5.1 a. Binary PA")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Mistakes")
        pyplot.savefig('51a Binary PA')
        pyplot.show()

    if type == "b.":
        # training accuracy curve, x is iteration, y is accuracy
        pyplot.plot([i for i in range(1, iteration+1)], accuracy)
        pyplot.title("5.1 b. Binary PA Training")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('51b Binary PA Training')
        pyplot.show()
        

        # testing accuracy curve
        pyplot.plot([i for i in range(1, iteration+1)], testAccuracy[:iteration])
        pyplot.title("5.1 b. Binary PA Testing")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('51b Binary PA Testing')
        pyplot.show()
        

def AveragePerceptron(iteration):
    mistake = [0] * iteration
    accuracy = [0] * iteration
    weightVector = np.array([0] * featureNum)
    weightSum = np.array([0 * featureNum])

    vote = 1

    for i in range(iteration):
        for j in range(dataLength):
            # prediction
            yt = np.sign(np.dot(weightVector, X_train[j]))

            # mistake and weight update
            if yt != labelDic[y_train[j]]:
                weightVector = weightVector + (np.dot(labelDic[y_train[j]], X_train[j]))
                weightSum = weightSum + np.dot(labelDic[y_train[j]] * vote, X_train[j])
                mistake[i] += 1

            else: 
                vote += 1

        # do the testing
        testing(weightVector, i)

        accuracy[i] = (dataLength - mistake[i]) / dataLength

    weightAverage = weightVector - weightSum / vote

    # testing accuracy curve
    pyplot.plot([i for i in range(1, iteration+1)], testAccuracy[:iteration])
    pyplot.title("5.1 c. Binary Averaged Perceptron Testing")
    pyplot.xlabel("Iterations")
    pyplot.ylabel("Accuracy")
    pyplot.savefig('51c Binary Averaged Perceptron Testing')
    pyplot.show()
    

if __name__ == "__main__":

    # 5.1 a.
    iteration = 50
    type = "a."
    BinaryPerceptron(iteration, type, False)
    PassiveAggressive(iteration, type, False)

    # 5.1 b.
    iteration = 20
    type = "b."
    BinaryPerceptron(iteration, type, True)
    PassiveAggressive(iteration, type, True)

    # 5.1 c.
    iteration = 20
    AveragePerceptron(iteration)

    #5.1 d.
    iteration = 20
    type = "d."
    BinaryPerceptron(iteration, type, True)
#%%