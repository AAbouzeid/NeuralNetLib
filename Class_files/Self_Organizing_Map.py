import numpy as np
import random as rand
import matplotlib.pyplot as plt
import utils
import copy


class SOM:
    """
    The SOM class includes the information about the Self-Organizing map
    The only functions that should be invoked externally are SOM.train() and SOM.test(), all the other
    internal functions are not supposed to be invoked and as a user you're not encouraged to run them, as doing so
    might produce unpredictable results.
    The class is expecting the inputData to be a python list of np.array objects of shape (1xn), with n being the
    dimension of the data.
    :param
    """
    def __init__( self, side=5, inputData=None, radius=5, numIters=10000, learningParam=0.5 ):
        self.side = side
        self.inputData = inputData
        self.weights = [ [0 for _ in range( side )] for _ in range( side ) ]
        self.radius = radius
        self.numIters = numIters
        self.learningParam = learningParam
        self.inputData = inputData
        self.density = np.zeros((np.shape(self.weights)[0:2]))
        self.initWeights()


    def initWeights(self):
        """
        This function initializes the weights of the SOM
        :return: Void
        """
        # Check if the user gave input/output data
        if not self.inputData:
            print(" Input/Output data were not provided!\n")
            return

        for i in range( self.side ):
            for j in range( self.side ):
                self.weights[i][j] = copy.deepcopy( utils.select_pattern( self.inputData ) )


    def euclidean(self, x, y):
        """
        This function given two vectors, computes the euclidean distance
        :param x: np array of shape (1xn)
        :param y: np array of shape (1xn)
        :return: Euclidean distance between x adn y
        """
        return np.linalg.norm( x - y )


    def findClosestNode( self, dataPoint, distFunc=euclidean ):
        """
        This function uses the distance according to a chosen distance function
        :param dataPoint: The point x in the data that we are trying to find the closest neuron to
        :param distFunc: A function object, corresponding to the distance metric chosen by the user.
        :return:
        """
        minDist = float( 'inf' )
        minPoint = ( 0, 0 )
        for i in range( self.side ):
            for j in range( self.side ):
                tempDist = distFunc( dataPoint, self.weights[i][j] )
                if tempDist < minDist:
                    minDist = tempDist
                    minPoint = np.array([ i, j ]).reshape( (2, 1) )

        return minPoint


    def updateNeuron( self, wVec, x, learningParam ):
        """
        This function provides the updated weight neuron according to the weight update formula:
        wj(t+1) = wj(t) + a(t) * ( x - wj(t) )
        :param learningParam: The time-decreasing learning parameter
        :param wVec: The vector wj(t)
        :param x: The input pattern
        :return: the vector wj(t+1)
        """
        return wVec*1.0 + learningParam * np.subtract( x, wVec )


    def updateNeighbor( self, radius, wj, dataPoint, distFunc=euclidean, learningParam=0.5 ):
        """
        This function will iterate over all the nodes in the lattic, and update the ones in the neighborhood of the
        chosen node, according to the update function indicated in updateNeuron() above.
        :param radius: a time decreasing radius to indicate the neighborhood that we will activate
        :param weightP: The coordinates of the neuron in the lattice that is the center of our neighborhood
        :param dataPoint: The pattern x in the inputData that we are considering
        :param distFunc: A function object, corresponding to the distance metric chosen by the user.
        """
        inRadius = 0
        for i in range( self.side ):
            for j in range( self.side ):
                if distFunc( wj, np.array([ i, j ]).reshape( (2, 1) ) ) < radius:
                    inRadius+=1
                    self.weights[i][j] = self.updateNeuron( self.weights[i][j], dataPoint, learningParam=learningParam )


    def mapPE(self, *output):
        """
        this function returns the densities of each PE
        """
        if output:
            mapOut = np.zeros((10,10,3))
        for i in range(len(self.inputData)):
            wj = self.findClosestNode(self.inputData[i], distFunc=self.euclidean)
            self.density[int(wj[0])][int(wj[1])] += 1
            if output:
                mapOut[int(wj[0])][int(wj[1])] = output[0][i]
        if output:
            return mapOut

    def train(self):
        """
        This function trains the SOM.
        """
        learnP = self.learningParam
        counter = 0
        for i in range( self.numIters ):
            radius = (self.numIters - i)/self.numIters * self.radius
            if not i % 500:
                learnP = ( 1.0 / ( 1.0 + 1.0 * counter ) ) * self.learningParam
                counter += 1
            if not i % 10000:
                self.mapPE()
                utils.densityPlot(self.density)
                utils.vectorPlot(self.side, self.weights)

            xk = utils.select_pattern( self.inputData )
            wj = self.findClosestNode( xk, distFunc=self.euclidean )
            self.updateNeighbor( radius, wj, xk, distFunc=self.euclidean, learningParam=learnP)

        self.mapPE()