import utils
import Network
import numpy as np

# Constants
A = 1
B = 0.2
maxTanh = 0.95
minTanh = -0.95
maxInpTanh = 0.8
minInpTanh = -0.8

# max and min for scaling updated during creation of data
maxX = -99999
minX = 99999
maxY = -99999
minY = 99999

# Creating the data set for training the network
trainPoints = []
trainDistP = []
for i in range( 200 ):
    trainPoints.append( utils.signal1( i, 1 ) )
    trainDistP.append( utils.distortFunc( trainPoints[-1], A, B ) )

    # update min and max for the actual signal (output)
    maxY = max( maxY, trainPoints[ -1 ] )
    minY = min( minY, trainPoints[ -1 ] )

    # update min and max for the distorted points (input)
    maxX = max( maxX, trainDistP[ -1 ] )
    minX = min( minX, trainDistP[ -1 ] )


# Creating the data set for during the test learning
testPoints = []
testDistP = []
for i in range( 200, 300 ):
    testPoints.append( utils.signal1( i, 1 ) )
    testDistP.append( utils.distortFunc( testPoints[-1], A, B ) )

    # update min and max for the actual signal (output)
    maxY = max(maxY, testPoints[-1])
    minY = min(minY, testPoints[-1])

    # update min and max for the distorted points (input)
    maxX = max(maxX, testDistP[-1])
    minX = min(minX, testDistP[-1])


# Scale the points ( outputs ) and distorted points ( inputs ) and construct the np arrays for training data
for i in range( 200 ):
    trainDistP[ i ] = np.array([1, utils.scaleNum( trainDistP[ i ], maxX, minX, minInpTanh, maxInpTanh )]).reshape((2, 1))
    trainPoints[ i ] = np.array([utils.scaleNum( trainPoints[ i ], maxY, minY, minTanh, maxTanh )]).reshape((1, 1))

# Scale the points ( outputs ) and distorted points ( inputs ) and construct the np arrays for test data
for i in range( 100 ):
    testDistP[ i ] = np.array([1, utils.scaleNum( testDistP[ i ], maxX, minX, minInpTanh, maxInpTanh )]).reshape((2, 1))
    testPoints[ i ] = np.array([utils.scaleNum( testPoints[ i ], maxY, minY, minTanh, maxTanh )]).reshape((1, 1))


# build and train the model
testNetwork3 = Network.Network(inputData=trainDistP, outputData=trainPoints, hiddenDim=5, inputDim=2, outputDim=1,
                               hidden=1, maxIter=500000, batchSize=0, outScale=[maxY, minY, minTanh, maxTanh],
                               inScale=[maxX, minX, minInpTanh, maxInpTanh], testData=testDistP, desiredData=testPoints )
testNetwork3.train()

# accuracy on Test data
maxDiff = 0
maxCouple = None
for i in range( len(testDistP) ):
    yActual = utils.unScale( testNetwork3.test( testDistP[i] ), [maxY, minY, minTanh, maxTanh])[0][0]
    yDesired = utils.unScale( testPoints[ i ], [maxY, minY, minTanh, maxTanh] )[0][0]
    temp = abs( yActual - yDesired )
    if temp > maxDiff:
        maxDiff = temp
        maxCouple = yActual, yDesired
    print( "Actual output by network is: ", yActual, " The input undistored signal is: ", yDesired)
print( "maxDiff: ", maxDiff)
print( "maxCouple: ", maxCouple)


inputSignal = []
desiredSignal = []
actualSignal = []

for idx in range(len(trainPoints)):
    actualSignal.append(utils.unScale(testNetwork3.test(trainDistP[idx]), [maxY, minY, minTanh, maxTanh]))
    desiredSignal.append(utils.unScale(trainPoints[idx], [maxY, minY, minTanh, maxTanh]))
    inputSignal.append(utils.unScale(trainDistP[idx][1], [maxY, minY, minTanh, maxTanh]))
inputSignal = [p.tolist()[0] for p in inputSignal]
desiredSignal = [p.tolist()[0] for p in desiredSignal]
actualSignal = [p.tolist()[0] for p in actualSignal]

utils.plotresults(testNetwork3.stepErr[0], testNetwork3.stepErr[1], testNetwork3.stepErr[2], desiredSignal, inputSignal,
                  actualSignal, None, None)

"""
========================================================================================================================
"""



# Part 2.2

# Testing with s1
s1DistP = []
s1Points = []

# Generating the points and distortion points for the s1 signal function
for i in range( 200 ):
    s1Points.append(utils.signalS1(i, 1))
    s1DistP.append(utils.distortFunc(s1Points[-1], A, B))

    # update min and max for the actual signal (output)
    maxY = max(maxY, s1Points[-1])
    minY = min(minY, s1Points[-1])

    # update min and max for the distorted points (input)
    maxX = max(maxX, s1DistP[-1])
    minX = min(minX, s1DistP[-1])

# scaling and generating input for the s1 signal
for i in range( 200 ):
    s1DistP[ i ] = np.array([1, utils.scaleNum( s1DistP[ i ], maxX, minX, minInpTanh, maxInpTanh )]).reshape((2, 1))
    s1Points[ i ] = np.array([utils.scaleNum( s1Points[ i ], maxY, minY, minTanh, maxTanh )]).reshape((1, 1))

# accuracy on s1 signal data
print("testing on the s1 signal data: ")
maxDiff = 0
maxCouple = None
for i in range( len(s1DistP) ):
    yActual = utils.unScale( testNetwork3.test( s1DistP[i] ), [maxY, minY, minTanh, maxTanh])[0][0]
    yDesired = utils.unScale( s1Points[ i ], [maxY, minY, minTanh, maxTanh] )[0][0]
    temp = abs( yActual - yDesired )
    if temp > maxDiff:
        maxDiff = temp
        maxCouple = yActual, yDesired
    print( "s1 data - Actual output by network is: ", yActual, " The input undistored signal is: ", yDesired)
print( "s1 data - maxDiff: ", maxDiff)
print( "s1 data - maxCouple: ", maxCouple)

inputSignal1 = []
desiredSignal1 = []
actualSignal1 = []

for idx in range(len(s1Points)):
    actualSignal1.append(utils.unScale(testNetwork3.test(s1DistP[idx]), [maxY, minY, minTanh, maxTanh]))
    desiredSignal1.append(utils.unScale(s1Points[idx], [maxY, minY, minTanh, maxTanh]))
    inputSignal1.append(utils.unScale(s1DistP[idx][1], [maxY, minY, minTanh, maxTanh]))
inputSignal1 = [p.tolist()[0] for p in inputSignal1]
desiredSignal1 = [p.tolist()[0] for p in desiredSignal1]
actualSignal1 = [p.tolist()[0] for p in actualSignal1]

utils.plotresults(None, None, None, desiredSignal1, inputSignal1, actualSignal1, None, None)

# Testing with s2

# Generating the points and distortion points for the s2 signal function
s2Points = utils.signalS2()
s2DistP = [ utils.distortFunc(point, A, B) for point in s2Points]

maxX = max( maxX, max(s2DistP) )
minX = min( minX, min(s2DistP) )
maxY = max( maxY, max(s2Points) )
minY = min( minY, min(s2Points) )

# scaling and generating input for the s1 signal
for i in range(200):
    s2DistP[i] = np.array([1, utils.scaleNum(s2DistP[i], maxX, minX, minInpTanh, maxInpTanh)]).reshape((2, 1))
    s2Points[i] = np.array([utils.scaleNum(s2Points[i], maxY, minY, minTanh, maxTanh)]).reshape((1, 1))

# accuracy on s1 signal data
print("testing on the s2 signal data: ")
maxDiff = 0
maxCouple = None
for i in range(len(s2DistP)):
    yActual = utils.unScale(testNetwork3.test(s2DistP[i]), [maxY, minY, minTanh, maxTanh])[0][0]
    yDesired = utils.unScale(s2Points[i], [maxY, minY, minTanh, maxTanh])[0][0]
    temp = abs(yActual - yDesired)
    if temp > maxDiff:
        maxDiff = temp
        maxCouple = yActual, yDesired
    print("s2 data - Actual output by network is: ", yActual, " The input undistored signal is: ", yDesired)
print("s2 data - maxDiff: ", maxDiff)
print("s2 data - maxCouple: ", maxCouple)

inputSignal2 = []
desiredSignal2 = []
actualSignal2 = []

for idx in range(len(s2Points)):
    actualSignal2.append(utils.unScale(testNetwork3.test(s2DistP[idx]), [maxY, minY, minTanh, maxTanh]))
    desiredSignal2.append(utils.unScale(s2Points[idx], [maxY, minY, minTanh, maxTanh]))
    inputSignal2.append(utils.unScale(s2DistP[idx][1], [maxY, minY, minTanh, maxTanh]))
inputSignal2 = [p.tolist()[0] for p in inputSignal2]
desiredSignal2 = [p.tolist()[0] for p in desiredSignal2]
actualSignal2 = [p.tolist()[0] for p in actualSignal2]

utils.plotresults(None, None, None, desiredSignal2, inputSignal2, actualSignal2, None, None)