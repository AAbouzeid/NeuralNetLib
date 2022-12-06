import Layer
import numpy as np
import copy
import utils


class Network:
    # TODO: The input data and output data are assumed to be scaled before given to the network class - add all scaling libs to the utils file
    # TODO: Add bias to hidden layers
    """
        The Network class includes the information about the network as a whole including information about it's layers
        such as the number of layers, and their respective dimensions.
        The only functions that should be invoked externally are network.train() and network.test(), all the other
        internal functions are not supposed to be invoked and as a user you're not encouraged to run them, as doing so
        might produce unpredictable results.
        :param hidden: Integer to indicate number of hidden layers wanted - defaults to 1
        :param batchSize: Integer to indicate the batch size
        :param inputData: Python List of np.array of numbers - the objects inside the list must be np.arrays of
        dimension (Nx1) even if the data points are just a single number.
        :param hiddenDim: Integer indicating the size of the hidden layer if known
        :param inputDim: Integer indicating the dimension of each training sample (Vector)
        :param outputDim: Integer indicating the dimension of the outputs (Vector)
        """

    def __init__(self, hidden=1, batchSize=0, inputData=None, outputData=None,
                 hiddenDim=0, inputDim=0, outputDim=0, maxIter=100000, inScale=[], outScale=[], testData=None,
                 desiredData=None, classification=False, alpha=0.7, learnP1=0.1, learnP2=0.1):
        self.hidden = hidden
        self.batchSize = batchSize
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.hiddenDim = hiddenDim + 1 if hiddenDim else self.computeHiddenDim()
        self.alpha = alpha

        self.inputData = inputData
        self.outputData = outputData
        self.inputLayer = Layer.InputLayer(size=inputDim)
        self.outputLayer = Layer.OutLayer(size=outputDim, alpha=self.alpha, learnP=learnP1)
        self.hiddenLayers = [Layer.HiddenLayer(size=hiddenDim, alpha=self.alpha, learnP=learnP2)]
        self.layers = [self.inputLayer] + self.hiddenLayers + [self.outputLayer]
        self.weights = self.initWeights()
        self.maxIter = maxIter
        self.setHiddenW()
        self.inScale = inScale
        self.outScale = outScale
        self.testData = testData
        self.stepErr = [[], [], [], [], []]
        self.desiredData = desiredData
        self.classification = classification
        self.bestWeight = None
        self.bestAccuracy = 0

    def setHiddenW(self):
        """
        This function sets the incoming/outgoing weights for the hidden layers
        :return: Void
        """
        # Setting the outgoing weight of the input layer
        self.inputLayer.outWeights = self.weights[0]

        # Setting the ingoing/outgoing weights for the hidden layers
        # print( len(self.weights), len(self.layers) - 1 )
        for i in range(1, len(self.layers) - 1):
            self.layers[i].inWeights = self.weights[i - 1]
            self.layers[i].outWeights = self.weights[i]
            self.layers[i].initSumChange()

        # Setting the ingoing weight for the output layer
        self.outputLayer.inWeights = self.weights[- 1]
        self.outputLayer.initSumChange()

    def computeHiddenDim(self):
        """
        This function uses two heuristics to decide the number of hidden layers neurons using two heuristics
        1 - The neurons in the hidden layers = (inputDim + outputDim) / 2
        2 - The neurons in the hidden layers = (2/3 * inputDim) + outputDim
        Using the above heuristics the number of HiddenDim is the average of the two heuristics
        :return: Integer hiddenDim: the number of neurons in the hidden layers
        """
        heuristic1 = int((self.inputDim + self.outputDim) / 2)
        heuristic2 = int(int(2 / 3 * self.inputDim) + self.outputDim)
        return int((heuristic1 + heuristic2) / 2)

    def initWeights(self):
        """
        This function creates the initial weight matrix
        :return: W = list of initialized weight matrices for each layer
        """

        W = []
        for i in range(1, len(self.layers)):
            W.append(np.random.uniform(-0.1, 0.1, (self.layers[i].size, self.layers[i - 1].size)))

        # print("init Weights: ", W)
        return W

    def feedForward(self, xk):
        """
        This function does the network-wide feedforward step, and mutates each layer's Y values and NET values.
        :param: xk --> input vector sample
        """
        W = self.weights

        # layer 0
        x = np.tanh(xk)

        # bias node
        x[0] = 1

        # compute output vector for each layer
        y_actual = []
        NETS = []
        for layer in range(len(W)):
            NET = np.matmul(W[layer], x)
            y = np.tanh(NET)

            # bias node
            if layer != len(W) - 1:
                y[0] = 1
                NET[0] = 0

            NETS.append(copy.deepcopy(NET))
            y_actual.append(np.array(y))
            x = y

        # print("Feedforward stats!: ")
        # Assign the Y val and nets of the input layer
        self.inputLayer.Y = np.array(xk).reshape((self.inputLayer.size, 1))
        self.inputLayer.NETs = np.array(xk).reshape((self.inputLayer.size, 1))
        # print("Input layer info: ", self.inputLayer.Y, self.inputLayer.NETs, "\n")

        # assign the Y val and nets of the hidden layers
        for layer in range(len(y_actual) - 1):
            hiddenTemp = self.hiddenLayers[layer]
            hiddenTemp.Y = np.array(y_actual[layer]).reshape((hiddenTemp.size, 1))
            hiddenTemp.NETs = np.array(NETS[layer]).reshape((hiddenTemp.size, 1))
            # print("hiddenTemp info: ", hiddenTemp.Y, hiddenTemp.NETs, "\n")

        # assign the Y val and nets of the output layer
        self.outputLayer.Y = np.array(y_actual[-1]).reshape((self.outputLayer.size, 1))
        self.outputLayer.NETs = np.array(NETS[-1]).reshape((self.outputLayer.size, 1))
        # print("Output layer info: ", self.outputLayer.Y, self.outputLayer.NETs, "\n")

    def test(self, xk):
        """
        This is the function that is used to test the network with a specific data point xk
        :param xk: testing data point - column vectorized
        :return: The output of the network - column vectorized
        """
        self.feedForward(xk)
        return self.outputLayer.Y

    def backProp(self, yk, batch=0):
        """
        This is the function that triggers the backpropagation changes for every layer.
        :param yk: The desired output
        :param batch: batch flag, to indicate whether it's batch backpropagation or online.
        :return: void
        """
        # back propagation on the output layer
        self.outputLayer.backProp(yk, self.hiddenLayers[-1].Y, batch)

        # back propagation for the hidden layers
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].backProp(self.layers[i + 1].delta, self.layers[i - 1].Y, batch)
        # print("Weights after backProp: ", self.weights)

    def train(self):
        """
        This is a high-level "train" function that determines if the training is online or batch, and calls the
        relevant functions accordingly.
        :return: void
        """
        # Check if the user gave input/output data
        if not self.inputData and self.outputData:
            print(" Input/Output data were not provided! ")
            return

        # if there is a batchSize given, train using batch training
        if self.batchSize:
            self.trainBatch()

        # otherwise, use online training
        else:
            self.trainOnline()

        if self.bestWeight:
            self.weights = self.bestWeight

    def trainOnline(self):
        """
        This function trains the network online by combining the feedforward/backprop functions defined above.
        :return: void
        """
        for i in range(self.maxIter):
            xk, yk = utils.select_pattern(self.inputData, self.outputData)
            self.feedForward(xk)
            self.backProp(yk)

            if not (i % (self.maxIter / 10)):
                if not self.classification:
                    self.getAccuracy(i)
                else:
                    self.classAccuracy(i)

    def updateWLayers(self):
        """
        This function updates the weight values for each layer for batch learning.
        :return: void
        """
        # update output layer
        self.outputLayer.updateW()

        # update hidden layers
        for hidd in self.hiddenLayers:
            hidd.updateW()

    def trainBatch(self):
        """
        This function trains the network batch by combining the feedforward/backprop functions defined above.
        :return: void
        """
        # iterate over num iterations divided by the batch size
        for i in range(int(self.maxIter / self.batchSize)):

            # iterate over number of batchsize, and aggregate the change in weights
            for j in range(self.batchSize):
                xk, yk = utils.select_pattern(self.inputData, self.outputData)
                self.feedForward(xk)
                self.backProp(yk, batch=1)

            self.updateWLayers()
            if not ((i * (j + 1)) % (self.maxIter / 10)):
                if not self.classification:
                    self.getAccuracy(i * j)
                else:
                    self.classAccuracy(i * j)

    def getAccuracy(self, step):
        """
        This function computes the average error for the network on training and test data at a specific step: step, and
        saves it for plotting.
        :return: void
        """
        print('Step:', str(step))
        self.stepErr[0].append(step)
        avgErr = 0

        # accuracy on training data
        hit = 0
        miss = 0
        for i in range(len(self.inputData)):
            xk = round(float(self.test(self.inputData[i])[0]))
            if self.outputData[i][0] == 1:
                if xk == 1:
                    hit += 1
                else:
                    miss += 1

        # Calculating the average of training data error
        avgErr = (hit / (hit + miss))*100
        print('  Training Classification Accuracy:', str(avgErr), '%\n')
        # print(avgErr)
        self.stepErr[1].append(avgErr)

        # accuracy on Test data
        for i in range(len(self.testData)):
            xk = utils.unScale(self.test(self.testData[i])[0], self.outScale)
            avgErr += (xk - utils.unScale(self.desiredData[i], self.outScale)[0]) ** 2

        # Calculating the average of test data error
        if self.bestAccuracy:
            if avgErr < self.bestAccuracy:
                self.bestAccuracy = avgErr
                self.bestWeight = copy.deepcopy(self.weights)

        self.stepErr[2].append(avgErr)

    def classAccuracy(self, step):
        """
        This function computes the percentage correctness of classification data at step: step and prints it to the
        terminal.
        :param step: Integer indicating which step of the learning the network is currently at
        :return: void
        """
        correct = 0
        if step < 30000:
            self.stepErr[0].append(step)

        # accuracy on training data
        for i in range(len(self.inputData)):
            yActual = utils.classify(self.test(self.inputData[i])) * 0.95
            if np.array_equal(yActual, self.outputData[i]):
                correct += 1

        perc = int((correct / len(self.inputData)) * 100)
        if step < 30000:
            self.stepErr[1].append(perc)
        # print( "percentage correct classification on training data at step: ", step, " is: ", perc  )

        correct = 0
        # accuracy on Test data
        for i in range(len(self.testData)):
            yActual = utils.classify(self.test(self.testData[i]))
            if np.array_equal(yActual, self.desiredData[i]):
                correct += 1

        perc = int((correct / len(self.testData)) * 100)
        if step < 30000:
            self.stepErr[2].append(perc)
        # print( "percentage correct classification on test data at step: ", step, " is: ", perc, "\n" )
        if correct > self.bestAccuracy:
            self.bestAccuracy = correct
            self.bestWeight = copy.deepcopy(self.weights)