import numpy as np

#TODO: Add momentum = 0, and always use momentum
class Layer:
    """
    The layer class contains the information about each layer that all other layers inherit form.
    :param size: to indicate the size of the layer
    :param NETS: a vector of dimension (size x 1) numpy array --> definition in slides
    :param Y: a vector of dimensions (size x 1) numpy array --> f(NETS) - s.t f is the activation function of choice
    NOTE: the NET of each node in the layer is going to be an (N x 1) numpy array
    """
    def __init__( self, size=1 ):
        self.size = size
        self.NETs = np.zeros( ( size, 1 ) )
        self.Y = np.zeros( ( size, 1 ) )

class InputLayer( Layer ):
    """
    The layer class contains the information about each layer
    :param hidden: to indicate whether this is a hidden layer or not
    :param size: to indicate the size of the layer
    NOTE: the NET of each node in the layer is going to be an (N x 1) numpy array
    """

    def __init__(self, size=1, outWeights=None):
        super( InputLayer, self ).__init__( size )
        self.outWeights = outWeights

class OutLayer( Layer ):
    """
    The Output layer class contains the information the output layer
    :param hidden: to indicate whether this is a hidden layer or not
    :param size: to indicate the size of the layer
    NOTE: the NET of each node in the layer is going to be an (N x 1) numpy array
    """

    def __init__(self, size=1, inWeights=None, learnP=0.1, delta=None, alpha=0.8):
        super( OutLayer, self ).__init__( size )
        self.inWeights = inWeights
        self.learnP = learnP
        self.delta = delta
        self.sumChange = None
        self.prevInW = None
        self.alpha = alpha

    def initSumChange(self):
        self.sumChange = np.zeros( self.inWeights.shape )
        self.prevInW = np.zeros( self.inWeights.shape )

    def backProp(self, desired, y_prev, batch=0):
        """
        This is the output layer backprop function, it supports both batch and online learning
        :param: desired: The vector of desired outputs of dimension ( size x 1 )
        :param: y_prev: The vector of the previous layer's Y values
        The implementation of backpropagation for the output layer. and mutates the incoming weights
        """
        if batch:
            self.sumChange = self.sumChange + self.learnP * np.matmul( self.delt( desired ), y_prev.transpose() )
        else:
            temp = self.learnP * np.matmul( self.delt( desired ), y_prev.transpose() ) + self.alpha * self.prevInW
            self.inWeights += temp
            self.prevInW = temp

    def delt(self, desired):
        """
        This function computes delta for output layer
        inputs:
            desired = vector of desired outputs in the output layer - dimensions --> (self.size x 1)
        output:
            delta = vector of the deltas for every node in the output layer - dimensions --> (self.size x 1)
        """
        delta = np.subtract(desired, self.Y)
        self.delta = np.multiply( delta, np.subtract( 1.0, np.square( np.tanh( self.NETs )  ) ) )
        return self.delta

    def updateW(self):
        """
        This function updates the weights for output layer.
        :return:
        """
        self.inWeights += self.sumChange + self.alpha * self.prevInW
        self.prevInW = self.sumChange + self.alpha * self.prevInW
        self.sumChange = np.zeros( ( self.size, 1 ) )

class HiddenLayer( Layer ):
    """
        The Hidden layer class contains the information about each Hidden layer
        :param hidden: to indicate whether this is a hidden layer or not
        :param size: to indicate the size of the layer
        NOTE: the NET of each node in the layer is going to be an (N x 1) numpy array
        """

    def __init__(self, size=1, inWeights=None, outWeights=None, learnP=0.1, delta=None, alpha=0.8):
        super( HiddenLayer, self ).__init__( size )
        self.inWeights = inWeights
        self.outWeights = outWeights
        self.learnP = learnP
        self.delta = delta
        self.sumChange = None
        self.alpha = alpha

    def initSumChange(self):
        self.sumChange = np.zeros( self.inWeights.shape )

    def delt(self, deltaN ):
        """
        This function computes delta for hidden layer
        inputs:
            deltaN = delta vector for the next layer
        output:
            delta = vector of the deltas for every node in currLayer layer
        """

        delta = np.matmul( self.outWeights.transpose(), deltaN )
        self.delta = np.multiply( delta, np.subtract( 1.0, np.square( np.tanh( self.NETs )  ) ) )

        return self.delta

    def backProp(self, deltaN, y_prev, batch=0):
        """
        This function does backprop for hidden layer, supporting both online and batch learning
        :param: desired: The vector of desired outputs of dimension ( size x 1 )
        :param: y_prev: The vector of the previous layer's Y values
        The implementation of backpropagation for the output layer. and mutates the incoming weights
        """
        if batch:
            self.sumChange += self.learnP * np.matmul( self.delt( deltaN ), y_prev.transpose() )
        else:
            self.inWeights += self.learnP * np.matmul( self.delt( deltaN ), y_prev.transpose() )

    def updateW(self):
        """
        This function updates weights for hidden layer.
        :return: void
        """
        self.inWeights += self.sumChange
        self.sumChange = np.zeros( self.inWeights.shape )
