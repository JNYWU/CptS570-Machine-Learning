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

testAccuracy = [0] * 10000

def testing(weightVector, iteration):
    testArgMax = float("-inf")
    bingo = 0

    for i in range(testLength):
        # predict
        for j in range(10):

            currentWeight = weightVector[j * featureNum : j * featureNum + featureNum]
            testArg = np.dot(currentWeight, X_test[j])

            if testArg > testArgMax:
                testArgMax = testArg
                yt = j

        # correct guess
        if yt == y_test[j]:
            bingo += 1

    testAccuracy[iteration] = bingo / testLength

def MultiClassPerceptron(iteration, type, isTesting):
    # for question a. b. and c.
    if type != "d.":
        weightVector = np.array([0] * featureNum * 10)
        mistake = [0] * iteration
        accuracy = [0] * iteration
        argMax = float("-inf")

        # for averaged perceptron
        weightSum = np.array([0 * featureNum])
        vote = 1

        for i in range(iteration):
            for j in range(dataLength):
                
                # prediction
                # 10 classes
                for k in range(10):

                    currentWeight = weightVector[k * featureNum : k * featureNum + featureNum]
                    arg = np.dot(currentWeight, X_train[j])

                    if arg > argMax:
                        argMax = arg
                        yt = k

                # mistake and weight update
                if yt != y_train[j]:

                    mistake[i] += 1

                    # for readability
                    weightGuess = weightVector[yt * featureNum : yt * featureNum + featureNum]
                    weightCorrect = weightVector[y_train[j] * featureNum : y_train[j] * featureNum + featureNum]

                    weightCorrect = weightCorrect + X_train[j]
                    weightGuess = weightGuess - X_train[j]

                    weightVector[yt * featureNum : yt * featureNum + featureNum] = weightCorrect
                    weightVector[y_train[j] * featureNum : y_train[j] * featureNum + featureNum] = weightGuess
                    
                    weightSum = weightSum + np.dot(y_train[j] * vote, X_train[j])

                else:
                    vote += 1      

            #do the testing
            if isTesting:
                testing(weightVector, i)

            accuracy[i] = (dataLength - mistake[i]) / dataLength

    # for qustion d.
    if type == "d.":
        temp = 600
        d = 100
        mistake = [0] * iteration
        accuracy = [0] * iteration
        argMax = float("-inf")

        for n in range(temp):
            weightVector = np.array([0] * featureNum * 10)

            for i in range(iteration):
                for j in range(d):
                    
                    # prediction
                    # 10 classes
                    for k in range(10):

                        currentWeight = weightVector[k * featureNum : k * featureNum + featureNum]
                        arg = np.dot(currentWeight, X_train[j])

                        if arg > argMax:
                            argMax = arg
                            yt = k

                    # mistake and weight update
                    if yt != y_train[j]:

                        mistake[i] += 1

                        # for readability
                        weightGuess = weightVector[yt * featureNum : yt * featureNum + featureNum]
                        weightCorrect = weightVector[y_train[j] * featureNum : y_train[j] * featureNum + featureNum]

                        weightCorrect = weightCorrect + X_train[j]
                        weightGuess = weightGuess - X_train[j]

                        weightVector[yt * featureNum : yt * featureNum + featureNum] = weightCorrect
                        weightVector[y_train[j] * featureNum : y_train[j] * featureNum + featureNum] = weightGuess       

            #do the testing
            if isTesting:
                testing(weightVector, i)

            d += 100

        #plot
        pyplot.plot([i for i in range(1, temp*100, 100)], testAccuracy[:temp])
        pyplot.title("5.2 d. Multi Class Perceptron Testing")
        pyplot.xlabel("Training Examples")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('52d Multi Perceptron Testing')
        pyplot.show()

    # plot
    if type == "a.":
        # learning curve
        pyplot.plot([i for i in range(1, iteration+1)], mistake)
        pyplot.title("5.2 a. Multi Class Perceptron")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Mistakes")
        pyplot.savefig('52a Multi Perceptron')
        pyplot.show()

    if type == "b.":
        # Training accuracy
        pyplot.plot([i for i in range(1, iteration+1)], accuracy)
        pyplot.title("5.2 b. Multi Class Perceptron Training")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('52b Multi Perceptron Training')
        pyplot.show()

        # Testing accuracy
        pyplot.plot([i for i in range(1, iteration+1)], testAccuracy[:iteration])
        pyplot.title("5.2 b. Multi Class Perceptron Testing")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('52b Multi Perceptron Testing')
        pyplot.show()

    if type == "c.":
        #weightAveraged = weightVector - weightSum / vote
        pyplot.plot([i for i in range(1, iteration+1)], testAccuracy[:iteration])
        pyplot.title("5.2 c. Multi Averaged Perceptron Testing")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('52c Multi Averaged Perceptron Testing')
        pyplot.show()


def MultiPassiveAggressive(iteration, type, isTesting):
    weightVector = np.array([0] * featureNum * 10)
    mistake = [0] * iteration
    accuracy = [0] * iteration
    argMax = float("-inf")

    for i in range(iteration):
        for j in range(dataLength):
            
            # prediction
            # 10 classes
            for k in range(10):

                currentWeight = weightVector[k * featureNum : k * featureNum + featureNum]
                arg = np.dot(currentWeight, X_train[j])

                if arg > argMax:
                    argMax = arg
                    yt = k

            # mistake and weight update
            if yt != y_train[j]:

                # for readability
                weightGuess = weightVector[yt * featureNum : yt * featureNum + featureNum]
                weightCorrect = weightVector[y_train[j] * featureNum : y_train[j] * featureNum + featureNum]

                tau = (1 - (np.dot( weightCorrect, np.dot(weightCorrect, X_train[j]) ) - np.dot( weightGuess, np.dot(weightGuess, X_train[j]) ) ) ) / (np.linalg.norm( np.dot(weightCorrect, X_train[j]) - np.dot(weightGuess, X_train[j]) ) ** 2)

                weightCorrect = weightCorrect + np.dot(tau, X_train[j])
                weightGuess = weightGuess - np.dot(tau, X_train[j])

                weightVector[yt * featureNum : yt * featureNum + featureNum] = weightCorrect
                weightVector[y_train[j] * featureNum : y_train[j] * featureNum + featureNum] = weightGuess

                mistake[i] += 1
            

        #do the testing
        if isTesting:
            testing(weightVector, i)

        accuracy[i] = (dataLength - mistake[i]) / dataLength

    if type == "a.":
        pyplot.plot([i for i in range(1, iteration+1)], mistake)
        pyplot.title("5.2 a. Multi Class PA")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Mistakes")
        pyplot.savefig('52a Multi PA')
        pyplot.show()
        
    if type == "b.":
        pyplot.plot([i for i in range(1, iteration+1)], accuracy)
        pyplot.title("5.2 b. Multi Class PA Training")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('52b Multi PA Training')
        pyplot.show()
        
    if type == "b.":
        pyplot.plot([i for i in range(1, iteration+1)], testAccuracy[:iteration])
        pyplot.title("5.2 b. Multi Class PA Testing")
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Accuracy")
        pyplot.savefig('52b Multi PA Testing')
        pyplot.show()


if __name__ == "__main__":

    # 5.2 a.
    iteration = 50
    type = "a."
    MultiClassPerceptron(iteration, type, False)
    MultiPassiveAggressive(iteration, type, False)

    # 5.2 b.
    iteration = 20
    type = "b."
    MultiClassPerceptron(iteration, type, True)
    MultiPassiveAggressive(iteration, type, True)

    # 5.2 c.
    iteration = 20
    type = "c."
    MultiClassPerceptron(iteration, type, True)
    
    # 5.2 d.
    iteration = 20
    type = "d."
    MultiClassPerceptron(iteration, type, True)

