from sklearn.cluster import KMeans
from sklearn import metrics
from keras.datasets import mnist
from metrics import acc
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1)) #-1 means we dont specify and let numpy figure out the correct dimension to make the other specified dimensions work
x = np.divide(x, 255.)
# 10 clusters
n_clusters = len(np.unique(y))
# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(x)
# Evaluate the K-Means clustering accuracy.
acc(y, y_pred_kmeans)

