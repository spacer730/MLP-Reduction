import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
import matplotlib.colors
import matplotlib as mpl
from astropy.io import fits
from astropy.table import Table

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR

np.random.seed(33)

#Load the data and reshape it in the form we want
sdata = Table.read('Data/Synthethic-galaxy-data-20000-samples.dat', format='ascii')
sdata = np.array(sdata)

u, g, r, i, Z = sdata['u'], sdata['g'], sdata['r'], sdata['i'], sdata['z']
u_noerr, g_noerr, r_noerr, i_noerr, Z_noerr = sdata['u_noerr'], sdata['g_noerr'], sdata['r_noerr'], sdata['i_noerr'], sdata['z_noerr']
z_redshift = sdata['z_redshift']
galaxy_type = sdata['type']

u_g, g_r, r_i, i_Z = u-g, g-r, r-i, i-Z
u_g_noerr, g_r_noerr, r_i_noerr, i_Z_noerr = u_noerr-g_noerr, g_noerr-r_noerr, r_noerr-i_noerr, i_noerr-Z_noerr

u_i, g_i, Z_i = u-i, g-i, Z-i
u_i_noerr, g_i_noerr, Z_i_noerr = u_noerr-i_noerr, g_noerr-i_noerr, Z_noerr-i_noerr

X = np.array([i, u_i, g_i, r_i, Z_i])
X = X.transpose()

X_noerr = np.array([i_noerr, u_i_noerr, g_i_noerr, r_i_noerr, Z_i_noerr])
X_noerr = X_noerr.transpose()

#Shuffle data and build training and test set (75% and 25%)
X_train, X_test, X_noerr_train, X_noerr_test, galaxy_type_train, galaxy_type_test, z_redshift_train, z_redshift_test = train_test_split(X, X_noerr, galaxy_type, z_redshift)

#Build an instance of the UMAP algorithm class and use it on the dataset
trans = umap.UMAP().fit(X_train, galaxy_type_train)
train_embedding = trans.embedding_
test_embedding = trans.transform(X_test)

#Converting the galaxy type names to numbers
unique_g_types = np.unique(galaxy_type)

def galaxy_type_map(galaxy_type):
    for i in range(len(unique_g_types)):
        if galaxy_type == unique_g_types[i]:
            return i

galaxy_type_numbers_train = [galaxy_type_map(galaxy) for galaxy in galaxy_type_train]
galaxy_type_numbers_test = [galaxy_type_map(galaxy) for galaxy in galaxy_type_test]

#Building a galaxy-type classifier using k-nearest neighbours on the embedded training data and on the original data
knn = KNeighborsClassifier().fit(train_embedding, galaxy_type_train)
knn2 = KNeighborsClassifier().fit(X_train, galaxy_type_train)
#Seeing how the trained knn algorithm performs on the test set for both the embedded data as well as the original data
knn.score(test_embedding, galaxy_type_test)
knn2.score(X_test, galaxy_type_test)

#Training knn regressor for redshift on both the embedded data as well as the original data
neigh = KNeighborsRegressor(n_neighbors=7, weights='distance')
neigh.fit(train_embedding, z_redshift_train)
z_test_pred = neigh.predict(test_embedding)
score_pred = neigh.score(test_embedding, z_redshift_test)

neigh2 = KNeighborsRegressor(n_neighbors=7, weights='distance')
neigh2.fit(X_train, z_redshift_train)
z_test_pred_2 = neigh2.predict(X_test)
score_pred_2 = neigh2.score(X_test, z_redshift_test)

#Training support vector regressor for redshift on both the embedded data as well as the original data
svr1 = SVR().fit(train_embedding, z_redshift_train)
z_test_pred_3 = svr1.predict(test_embedding)
score_pred_3 = svr1.score(test_embedding, z_redshift_test)

svr2 = SVR().fit(X_train, z_redshift_train)
z_test_pred_4 = svr2.predict(X_test)
score_pred_4 = svr2.score(X_test, z_redshift_test)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#Split the dataset into two different groups
split = 1.3#np.mean(z_test_pred)+np.var(z_test_pred)**0.5
split_a = z_test_pred_2<split
split_b = z_test_pred_2>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(test_embedding[:, 0][split_a], test_embedding[:, 1][split_a], 3, c=z_test_pred_2[split_a], cmap='summer', norm=matplotlib.colors.LogNorm(vmin=0.01,vmax=split))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(test_embedding[:, 0][split_b], test_embedding[:, 1][split_b], 3, c=z_test_pred_2[split_b], cmap='autumn_r', norm=matplotlib.colors.Normalize(vmin=split,vmax=np.max(z_redshift)))#, norm=matplotlib.colors.LogNorm(vmin=split,vmax=np.max(zspec_perm)))
cbarb = fig.colorbar(CSb)
cbara.set_label('z<1.3')
cbarb.set_label('1.3<=z')
plt.show()

"""
#Galaxy type map
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]})

cmap = plt.cm.jet# define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0, 8, 9)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

CS = axs[0].scatter(train_embedding[:, 0], train_embedding[:, 1], 10, c=galaxy_type_numbers_train, cmap=cmap)

cbar = mpl.colorbar.ColorbarBase(axs[1], cmap=cmap, norm=norm, spacing='proportional', ticks=bounds+0.5, boundaries=bounds, format='%1i')
cbar.set_ticklabels(unique_g_types)
plt.show()
"""
