import numpy as np
import random
import Preprocess as preprocess
import displayRecognition
from Class_files import Network
from Class_files import utils

# Using a utility function to flatten the images of the 40 subjects that are in the Facial Data set repo
flat_images = preprocess.pgmread(40)

# Downsampling step to reduce the image in size by 1/4
for i in range(len(flat_images)):
    if (i == 0):
        downsample = np.array([utils.downSample(flat_images[i])])
    else:
        downsample = np.concatenate((downsample, np.array([utils.downSample(flat_images[i])])), axis=0)

# convert the downsamples image to their PCA vectors
pca_img = preprocess.princomp(downsample, 252)

# choose a random subject from 11 to 40 to be our chosen person to recognize
subject_to_recognize = random.randint(11, 40)

# configure the pca data by doing the necessary scaling to be able to be used by the Neural Network
input_data, output_data = preprocess.configureData(pca_img, subject_to_recognize)

# Creating the training/Test daya input and output splits
train_in = []
train_out = []
test_in = []
test_out = []
for subject in range(28):
    img_start = (subject) * 9
    img_end = (subject + 1) * 9
    separate = random.sample(range(img_start, img_end), 9)
    for idx in separate[:5]:
        train_in.append(input_data[idx])
        train_out.append(output_data[idx])
    for idx in separate[5:]:
        test_in.append(input_data[idx])
        test_out.append(output_data[idx])

# Creating the network and training it
outscale = [np.amax(flat_images), np.amin(flat_images), -0.9, 0.9]
testNetwork = Network.Network(inputData=train_in,outputData=train_out,hiddenDim=10,inputDim=np.shape(input_data)[1],outputDim=1,
                              hidden=1,maxIter=30000,batchSize=0,outScale=outscale,inScale=outscale,testData=test_in,desiredData=test_out)
testNetwork.train()

# Collecting the output of the network
act_out = []
for img in range(len(test_in)):
    act_out.append(round(float(testNetwork.test(test_in[img]))))

# Collecting the results if True positives, True Negatives, False Positives, False Negatives
[TP, TN, FP, FN] = utils.results(act_out, test_out)
print('True Positive:', TP, '\nTrue Negative:', TN, '\nFalse Positive:', FP, '\nFalse Negative:', FN)

# Displaying the results visually for reference.
displayRecognition.recognizeImage(flat_images, input_data, subject_to_recognize, testNetwork)
utils.gridSearchHyperParameter(train_in, train_out, test_in, test_out, flat_images, input_data)