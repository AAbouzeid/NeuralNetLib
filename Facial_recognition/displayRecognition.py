import matplotlib.pyplot as plt


def recognizeImage(flat_images, input_data, subject_to_recognize, testNetwork):
    """
    This method shows the results of the training. It renders the image of the right person in color, and everyone else
    in grey.
    :param flat_images: an np.array of the flattened images
    :param input_data: list of pca images
    :param subject_to_recognize: index of the person to recognize by the network
    :param testNetwork: the network object that was trained.
    :return:
    """

    # The image of the person to recognize
    recognize_image = flat_images[(subject_to_recognize-11)*9].reshape(480, 640)

    # Showing the image of the person to recognize
    plt.imshow(recognize_image)
    plt.title('Subject to Recognize')
    plt.show()

    # iterate over the other subjects and if they are of the same person as the recognize_image they colored
    # otherwise they are in gray scale.
    for subject in range(int(len(flat_images)/9)):
        plt.subplot(5, 6, subject + 1)
        if round(float(testNetwork.test(input_data[subject * 9 + 8]))) == 1:
            plt.imshow(flat_images[subject * 9 + 8].reshape(480, 640))
        else:
            plt.imshow(flat_images[subject * 9 + 8].reshape(480, 640), cmap='gray')
        plt.axis('off')
    plt.show()