import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Class_files import utils


def pgmread(last_subject):
    """
    This function reads the images from the dataSet, and flattens them.
    :param last_subject: The idx of the last person to load
    :return: flat_images - an np.array of the flattened images.
    """

    # Iterate over the people in the subjects list
    for person in range(11, last_subject):
        directory = 'Facial_Dataset/yaleB' + str(person)
        print('Loading Data for Subject', str(person))

        # Ignoring person 14
        if (person == 14):
            continue

        # Iterating over the different view points for every person
        for viewpoint in range(9):
            file = directory + '/yaleB' + str(person) + '_P0' + str(viewpoint) + 'A+000E+00.pgm'
            img = mpimg.imread(file)
            if (person == 11) and (viewpoint == 0):
                flat_images = np.array([img.flatten()])
            else:
                flat_images = np.concatenate((flat_images, np.array([img.flatten()])), axis=0)

    n_img, pix = np.shape(flat_images)
    print('\nTotal Number of Images:', str(n_img))
    print('Pixels Per Image:', str(pix))
    return flat_images


def princomp(flat_images, n_components):
    scaled_images = StandardScaler().fit_transform(flat_images)
    pca = PCA(n_components=n_components)
    reduced_images = pca.fit_transform(scaled_images)
    n_img, princomp = np.shape(reduced_images)
    recover_images = pca.inverse_transform(reduced_images)

    print('\nPrinciple Components Per Image:', str(princomp))
    plt.grid()
    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.title('Principal Component Analysis')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance %')
    plt.show()

    return reduced_images


def configureData(reduce_img, recog_subject):
    config_inp = []
    config_out = np.zeros((len(reduce_img), 1, 1))
    for j in range(len(reduce_img)):
        config_inp.append(
            np.append([1], utils.scaleNum(reduce_img[j], np.amax(reduce_img), np.amin(reduce_img), -0.9, 0.9)).reshape(
                len(reduce_img[0]) + 1, 1))
        if recog_subject == math.ceil((j + 1) / (9)) + 10:
            config_out[j] = 1

    return config_inp, config_out