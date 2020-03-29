import numpy as np
import matplotlib
import pylab
import math
from skimage import img_as_float
from skimage.io import imread
from sklearn.cluster import KMeans

image = imread('/Users/nikitabaramiya/Desktop/ML/parrots.jpg')
#pylab.imshow(image)

data = img_as_float(image)

x, y, z = data.shape
objects_features_matrix = np.reshape(data, (x * y, z))

centers = [[] for i in range(21)]
predictions = [[] for i in range(21)]

for i in range(1, 21):
    kmeans = KMeans(init='k-means++', n_clusters=i, random_state=241)

    kmeans.fit(objects_features_matrix)
    predictions[i] = kmeans.predict(objects_features_matrix)

    centers[i] = kmeans.cluster_centers_

PSNR = 20 * math.log10(255) - 10 * math.log10(MSE)

# it is not my strong suit :(
