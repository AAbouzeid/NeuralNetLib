from sklearn.decomposition import PCA
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import random

# Image extraction
Trainfilename = "/Users/aabouzeid/Desktop/Senior_year/Spring22/Comp502/final_Project/yalefaces/subject01.gif"
img1 = img.imread(Trainfilename)
img = img.imread(Trainfilename).flatten()
listt = [ img ]
for i in range(499):
    listt.append( np.array( [ random.randint(0, 255) for _ in range(len(img))]) )

listImgs = np.array( listt ).reshape( ( 500, len( listt[ 0 ] ) ) )
# print(listImgs)
# print(listImgs.shape)

k = 500
pca_10 = PCA(n_components=k)
mnist_pca_10_reduced = pca_10.fit_transform( listImgs )
# print( mnist_pca_10_reduced.shape )
mnist_pca_10_recovered = pca_10.inverse_transform(mnist_pca_10_reduced)
# print( mnist_pca_10_recovered.shape )

image_pca_10 = mnist_pca_10_recovered[0].reshape( img1.shape )
# print( np.subtract(img1, image_pca_10))
plt.imshow(image_pca_10, cmap='gray_r')
plt.title('Compressed image with ' + str(k) + ' components', fontsize=15, pad=15)
plt.savefig("image_pca_"+str(k)+".png")