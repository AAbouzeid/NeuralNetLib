from math import exp as e
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import math
from collections import deque
import Network

SECONDS = 10

# util functions
def select_pattern(X, Y):
    """
    inputs:
        X = list of input vectors - np.arrays
        Y = list of output vectors - np.arrays
    output:
        xk = input vector
        yk = output vector
    """
    rand_pattern = random.randint(0, len(X) - 1)
    return X[rand_pattern], Y[rand_pattern]


def plotresults(learning_steps, train_err, test_err, des_out, train_in, act_train_out, test_in, act_test_out):
    """
    inputs:
        learning_steps = number of steps the algorithm has taken
        train_err = mean absolute error in the training set at each step
        test_err = mean absolute error in the testing set at each step
        des_out = desired output of the function at x
        train_in = input values (x) for training set
        act_train_out = actual output of the training set at x
        test_in = input values (x) for the testing set
        act_test_out = actual output of the testing set at x
    """
    if not (learning_steps is None):
        plt.subplot(2, 1, 1)
        plt.plot(learning_steps, train_err, linestyle='', marker='+', color='b', label='training error')
        if not (test_in is None):
            plt.plot(learning_steps, test_err, linestyle='', marker='x', color='r', label='testing error')
        plt.title('Learning History')
        plt.xlabel('Learning Steps')
        plt.ylabel('Mean Square Error')
        plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_in, color='red', label='distorted input')
    plt.plot(des_out, color='blue', label='desired output')
    plt.plot(act_train_out, linestyle='--', color='lightgreen', label='actual output (testing)')
    if not (test_in is None):
        plt.plot(test_in, act_test_out, linestyle='', marker='x', color='g', markersize=5, label='actual output (testing)')
    plt.title('Signal Processing for s2(nT) = random')
    plt.xlabel('Time (n)')
    plt.ylabel('Signal (s(nT))')
    plt.legend()

    plt.tight_layout()
    plt.draw()  # draw the plot
    plt.pause(5)  # show it for 5 seconds


def plotLearning(learning_steps, train_err, test_err):
    """
    inputs:
        learning_steps = number of steps the algorithm has taken
        train_err = mean absolute error in the training set at each step
        test_err = mean absolute error in the testing set at each step
        des_out = desired output of the function at x
        train_in = input values (x) for training set
        act_train_out = actual output of the training set at x
        test_in = input values (x) for the testing set
        act_test_out = actual output of the testing set at x
    """
    plt.plot(learning_steps, train_err, label='training')
    plt.title('Learning History')
    plt.xlabel('Learning Steps')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.show()


def plotClassification(desired, y, datatype):
    """
    :param desired: desired classification (list of lists)
    :param y: actual classification (list of lists)
    :param datatype: string of either 'train' or 'test'
    :return:
    """

    sample = []
    hit = []
    mis = []
    for i in range(len(desired)):
        if (desired[i] == [[1], [0], [0]]).all():
            sample.append(1)
            if (desired[i] == y[i]).all():
                hit.append(1)
                mis.append(None)
            else:
                hit.append(None)
                mis.append(1)

        elif (desired[i] == [[0], [1], [0]]).all():
            sample.append(2)
            if (desired[i] == y[i]).all():
                hit.append(2)
                mis.append(None)
            else:
                hit.append(None)
                mis.append(2)

        elif (desired[i] == [[0], [0], [1]]).all():
            sample.append(3)
            if (desired[i] == y[i]).all():
                hit.append(3)
                mis.append(None)
            else:
                hit.append(None)
                mis.append(3)

    plt.figure()
    plt.plot(sample,'o',markersize=10,color='black')
    plt.plot(hit,'o',markersize=5,color='green')
    plt.plot(mis,'o',markersize=5,color='red')
    plt.title('Target vs Recalled Value for Iris ' + datatype + 'ing Data')
    plt.xlabel('Index of Sample')
    plt.ylabel('Classification')
    plt.yticks([1,2,3], ['Setosa','Versacolor','Virginica'])
    plt.draw()  # draw the plot
    plt.pause(5)  # show it for 5 seconds

    plt.show()


def plotClasAccur(steps, acc_train, acc_test):
    """
    :param steps: number of steps (list)
    :param acc_train: accuracy of training data at each step
    :param acc_test: accuracy of testing data at each step
    :return:
    """
    plt.figure()
    plt.plot(steps, acc_train, '-', markersize=10, color='red', label='training accuracy')
    plt.plot(steps, acc_test, '-', markersize=10, color='blue', label='testing accuracy')
    plt.title('Learning Accuracy History')
    plt.xlabel('Learning Steps')
    plt.ylabel('Classification Accuracy (%)')
    plt.legend()
    plt.show()


def scale( k, lwr_bnd, upr_bnd ):
    """
    :param k: list of numpy arrays - for now of size 1 for scalar inputs
    :param lwr_bnd: lower bound
    :param upr_bnd: upper bound
    :return: scaled array
    """
    return ( (k - min(k)) / (max(k) - min(k))) * (upr_bnd - lwr_bnd) + lwr_bnd

def scaleNum( k, kMax, kMin, lwr_bnd, upr_bnd ):
    """
    :param k: the value to scale as a real number
    :param lwr_bnd: the l
    :param upr_bnd:
    :return:
    """
    return ( (k - kMin) / (kMax - kMin) ) * (upr_bnd - lwr_bnd) + lwr_bnd

def unScale( k, kMax, kMin, lwr_bnd, upr_bnd ):
    """
    Function unscales from tabh to real value
    :param dk:
    :param fmax:
    :param fmin:
    :param Dmax:
    :param Dmin:
    :return:
    """
    return ( k - lwr_bnd ) * ( (kMax - kMin)/ (upr_bnd - lwr_bnd)) + kMin

def unScale( k, lstBnds ):
    kMax, kMin, lwr_bnd, upr_bnd = lstBnds[0], lstBnds[1], lstBnds[2], lstBnds[3]
    """
    Function unscales from tabh to real value
    :param dk:
    :param fmax:
    :param fmin:
    :param Dmax:
    :param Dmin:
    :return:
    """
    return ( k - lwr_bnd ) * ( (kMax - kMin)/ (upr_bnd - lwr_bnd)) + kMin

def parse(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    inp = []
    out = []
    for i in range(1, len(lines)):
        inp.append(lines[i].split(" "))
        inp[-1] = [float(x) for x in inp[-1]]
        out.append(inp[-1][0])
    inp = np.array(inp)
    #inp = [np.array(x).reshape((len(x), 1)) for x in inp]
    #out = [np.array(x) for x in out]
    return inp, out

def parse2(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    out = []
    for i in range(1, len(lines)):
        out.append(lines[i].split(" "))
        out[-1] = [float(x) for x in out[-1]]
    return out


def parse3(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    inp = []
    out = []
    for i in range(len(lines)):
        if not i%2:
            inp.append(lines[i].split("  ")[1:])
            inp[-1][-1] = inp[-1][-1][:-1]
            inp[-1] = [float(x) for x in inp[-1]]
        else:
            out.append(lines[i].split("   ")[1:])
            out[-1][-1] = out[-1][-1][:-1]
            out[-1] = [float(x) for x in out[-1]]
    return inp, out


def parse4(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    output = []
    for i in range(1, len(lines)):
        output.append( round( float( lines[i] ), 8 ) )

    return output

def parse5(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    inp = []
    out = []
    for i in range(len(lines)):
        if not i%2:
            inp.append(lines[i].split("  ")[1:])
            inp[-1][-1] = inp[-1][-1][:-1]
            inp[-1] = [float(x) for x in inp[-1]]
        else:
            out.append(lines[i].split("   ")[1:])
            out[-1][-1] = out[-1][-1][:-1]
            out[-1] = [float(x) for x in out[-1]]
    input = [np.array(x) for x in inp]
    output = [np.array(x) for x in out]
    return input, output


def classify(output):
    maxIndex = np.where(output == output.max())
    for idx in range(len(output)):
        if idx == maxIndex[0][0]:
            output[idx] = 1
        else:
            output[idx] = 0
    return output


def confusionMatrix(trueClass, predClass):
    setSet = 0
    setVer = 0
    setVir = 0
    verSet = 0
    verVer = 0
    verVir = 0
    virSet = 0
    virVer = 0
    virVir = 0
    for i in range(len(trueClass)):
        if (trueClass[i] == [[1], [0], [0]]).all():
            if (predClass[i] == [[1], [0], [0]]).all():
                setSet += 1
            elif (predClass[i] == [[0], [1], [0]]).all():
                setVer += 1
            elif (predClass[i] == [[0], [0], [1]]).all():
                setVir += 1
        elif (trueClass[i] == [[0], [1], [0]]).all():
            if (predClass[i] == [[1], [0], [0]]).all():
                verSet += 1
            elif (predClass[i] == [[0], [1], [0]]).all():
                verVer += 1
            elif (predClass[i] == [[0], [0], [1]]).all():
                verVir += 1
        elif (trueClass[i] == [[0], [0], [1]]).all():
            if (predClass[i] == [[1], [0], [0]]).all():
                virSet += 1
            elif (predClass[i] == [[0], [1], [0]]).all():
                virVer += 1
            elif (predClass[i] == [[0], [0], [1]]).all():
                virVir += 1
    lab0 = ['True Class', 'Predicted Class']
    lab1 = ['Setosa', 'Versacolor', 'Virginica']
    df = pd.DataFrame([[setSet, setVer, setVir], [verSet, verVer, verVir], [virSet, virVer, virVir]],
                      index=pd.MultiIndex.from_tuples([(lab0[0], lab1[0]), (lab0[0], lab1[1]), (lab0[0], lab1[2])]),
                      columns=pd.MultiIndex.from_tuples([(lab0[1], lab1[0]), (lab0[1], lab1[1]), (lab0[1], lab1[2])]))
    df.columns.name = 'Predicted Class'
    df.index.name = 'True Class'
    print(df)
    accuracy = ((setSet + verVer + virVir) / len(trueClass)) * 100
    return accuracy


def tabulateFolds(foldAcc):
    labelRow = ['Train Accuracy (%)', 'Test Accuracy (%)']
    labelCol = []
    for idx in range(len(foldAcc)):
        labelCol.append('Fold ' + str(idx + 1))
    labelCol.extend(['Average', 'Std Dev'])
    trainAcc = [foldAcc[0][0], foldAcc[1][0], foldAcc[2][0]]
    trainAcc.extend([round(np.mean(trainAcc), 1), round(np.std(trainAcc), 3)])
    testAcc = [foldAcc[0][1], foldAcc[1][1], foldAcc[2][1]]
    testAcc.extend([round(np.mean(testAcc), 1), round(np.std(testAcc), 3)])
    df = pd.DataFrame([trainAcc, testAcc], index=labelRow, columns=labelCol)
    pd.set_option("display.max_columns", None)
    print(df)


def distortFunc( point, A, B ): return (A*point) + B*(point**2)


def signal1( n, T ): return 2.0 * math.sin( ( 2.0 * math.pi * n * T )/20.0 )


def signalS1( n, T ): return 0.8 * math.sin( ( 2.0 * math.pi * n * T )/10.0 ) + 0.25 * math.cos( ( 2.0 * math.pi * n * T )/25.0 )


def signalS2( mean=0.0, var=1.0, size=200 ): return np.random.normal( loc=mean, scale=var, size=( 1, size) ).tolist()[0]


def downSample( npArrImg ):
    outArr = deque()
    for i in range( npArrImg.size ):
        if (not i % 4):
            continue
        else:
            outArr.append( npArrImg[i] )

    return np.array( list(outArr) )


def results(act_out, des_out):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(act_out)):
        if act_out[i] == 1:
            if int(des_out[i][0]) == 1:
                TP += 1
            else:
                FP += 1
        else:
            if int(des_out[i][0]) == 1:
                FN += 1
            else:
                TN += 1
    return [TP, TN, FP, FN]

def gridSearchHyperParameter( train_in, train_out, test_in, test_out, flat_images, input_data ):
    outscale = [np.amax(flat_images), np.amin(flat_images), -0.9, 0.9]
    resFile = "gridSearch.txt"
    res = open( resFile, "w")
    for i in range(1, 10):
        for j in range(1, 10):
            learnP = i*1.0/10
            alpha = j*1.0/10
            testNetwork = Network.Network(inputData=train_in, outputData=train_out, hiddenDim=10,
                                          inputDim=np.shape(input_data)[1], outputDim=1,
                                          hidden=1, maxIter=30000, batchSize=0, outScale=outscale, inScale=outscale,
                                          testData=test_in, desiredData=test_out, learnP1=learnP, alpha=alpha);
            testNetwork.train()
            act_out = []
            for img in range(len(test_in)):
                act_out.append(round(float(testNetwork.test(test_in[img]))))
            result = results(act_out, test_out)
            strr = "For LearnP = " + str(learnP) + " and alpha = " + str(alpha) + " - TP: " + str(result[0]) + " TN: " + str(result[1]) + " FP: " + str(result[2]) + " FN: " + str(result[3]) + "\n"
            res.write(strr)


# ******************** Helper Functions ******************** #
def vecToMat( vec ):
    out = []
    for i in range( 0, 57, 8):
        out.append( vec[i:i+8] )
    return out


def imageToVec( image ):
    """
    This function takes in an image and generates a list of np.arrays (64x1) vectors in order from top left to bottom
    right. (the vectors themselves are an 8x8 sub-image also ordered from top left to bottom right.)
    :param image: List of lists representative of the image pixel values
    :return: List of np.arrays
    """
    maxX = -float('inf')
    minX = float('inf')
    outList = []
    for row in range( 0, len( image ) - 8, 8 ):
        for col in range( 0, len( image[0] ) - 8, 8 ):
            vec = []
            for i in range( row, row + 8 ):
                for j in range( col, col + 8 ):
                    maxX = max( maxX, image[i][j] )
                    minX = min( minX, image[i][j] )
                    vec.append( image[i][j] )
            if len(vec) != 64:
                print( " len error wtf! " )
            outList.append( np.array( vec ).reshape( ( len( vec ), 1 ) ) )

    return maxX, minX, outList


def vecToImage( listVecs, lenRows, lenCols ):
    """
    This function takes in a list of vectors representing the image, and returns the list of lists representation of the
    image.
    :param listVecs: List of vectors, representing the 64-sized vector representation of the image.
    :return: list of lists, as the image representation
    """
    listVecs = [x.reshape(x.shape[1], x.shape[0]).tolist()[0] for x in listVecs]
    outList = [ [0] * lenCols for _ in range(lenRows) ]
    for row in range(0, lenRows - 8, 8):
        for col in range(0, lenCols - 8, 8):
            mat = vecToMat( listVecs[0] )
            for i in range( 8 ):
                outList[ row + i ][ col: col+8 ] = mat[ i ]
            listVecs.pop(0)

    return outList


def hiddenDimCalc( CR, N, n=64):
    """
    This function outputs an approximate size of the hidden dimension for a compression rato CR
    :param CR: The compression ratio
    :param N: Number of 8 x 8 blocks in the image
    :param n: the input dimension
    :return: The dimension of the hidden layer for the suggested compression ratio.
    """
    numerator = ( CR * 1.0 )*( N * 1.0 )
    denominator = 2.0 + ( ( N * 1.0 ) / ( n * 1.0 ) )
    return round( numerator / denominator )


def scaleVecs( listVecs, inScale ):
    out = []
    for vec in listVecs:
        vec2 = []
        for i in range( len( vec ) ):
            vec2.append(scaleNum( vec[ i ][0], inScale[ 0 ], inScale[ 1 ], inScale[ 2 ], inScale[ 3 ] ) )
        out.append( np.array( vec2 ).reshape( vec.shape ) )

    return out


def unscaleVecs( listVecs, inScale ):
    out = []
    for vec in listVecs:
        vec2 = []
        for i in range(len(vec)):
            vec2.append(unScale(vec[i][0], inScale))
        out.append(np.array(vec2).reshape(vec.shape))

    return out

def plot3( plot1, plot2, plot3 ):
    """
    inputs:
        learning_steps = number of steps the algorithm has taken
        train_err = mean absolute error in the training set at each step
        test_err = mean absolute error in the testing set at each step
        des_out = desired output of the function at x
        train_in = input values (x) for training set
        act_train_out = actual output of the training set at x
        test_in = input values (x) for the testing set
        act_test_out = actual output of the testing set at x
    """
    plt.plot( plot1, color='red', label='BP Output')
    plt.plot( plot2, color='blue', label='Desired Output')
    plt.plot( plot3, linestyle='--', color='lightgreen', label='Linear Regression Output')
    plt.title('BP vs Desired vs Linear Regression')
    plt.xlabel('island number')
    plt.ylabel('number of plant species')
    plt.legend()

    plt.tight_layout()
    plt.draw()  # draw the plot
    plt.pause(20)  # show it for 5 seconds


def latticePlot(trainingPoints, SOMweights):
    data = []
    for i in range(len(SOMweights)):
        for j in range(len(SOMweights)):
            data.append(SOMweights[i][j])

    latitude = SOMweights
    longitude = []
    for i in range(len(latitude)):
        line = []
        for j in range(len(latitude[0])):
            line.append(latitude[j][i])
        longitude.append(line)

    plt.scatter([x[0] for x in trainingPoints], [x[1] for x in trainingPoints], s=2, c='orange')
    plt.scatter([x[0] for x in data], [x[1] for x in data], c='b')
    for i in range(len(latitude)):
        plt.plot([x[0] for x in latitude[i]], [x[1] for x in latitude[i]], c='b')
    for j in range(len(longitude)):
        plt.plot([x[0] for x in longitude[j]], [x[1] for x in longitude[j]], c='b')
    plt.title('SOM')
    plt.show()


def densityPlot(densityMatrix):
    plt.pcolormesh(densityMatrix, cmap='inferno')
    plt.colorbar()
    plt.title('Density Graph')
    plt.show()

def vectorPlot(side, weights):
    fig, ax = plt.subplots(side, side)
    plt.suptitle('SOM Vector Visualization')
    for i in range(side):
        for j in range(side):
            ax[i, j].plot(weights[i][j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def vectorPlotClassify(side, weights, mapOut):
    fig, ax = plt.subplots(side, side)
    plt.suptitle('SOM Vector Visualization')
    for i in range(side):
        for j in range(side):
            ax[i, j].plot(weights[i][j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if mapOut[i, j].tolist() == [1, 0, 0]:
                ax[i, j].set_facecolor('red')
            elif mapOut[i, j].tolist() == [0, 1, 0]:
                ax[i, j].set_facecolor('blue')
            elif mapOut[i, j].tolist() == [0, 0, 1]:
                ax[i, j].set_facecolor('yellow')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()