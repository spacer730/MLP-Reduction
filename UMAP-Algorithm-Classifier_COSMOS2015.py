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

#Load the data en reshape it in the form we want
print('Openining the data file')
fdata = fits.open('Data/COSMOS2015_Laigle+_v1.1_removed_optical_nir_magnitudes_larger_90.fits')[1].data
print('File is open')

u, B, V, r, ip, zpp = fdata['u_MAG_AUTO'], fdata['B_MAG_AUTO'], fdata['V_MAG_AUTO'], fdata['r_MAG_AUTO'], fdata['ip_MAG_AUTO'], fdata['zpp_MAG_AUTO']
yHSC, Y, J, H, Ks = fdata['yHSC_MAG_AUTO'], fdata['Y_MAG_AUTO'], fdata['J_MAG_AUTO'], fdata['H_MAG_AUTO'], fdata['Ks_MAG_AUTO']
ch1, ch2, ch3, ch4 = fdata['SPLASH_1_MAG'], fdata['SPLASH_2_MAG'], fdata['SPLASH_3_MAG'], fdata['SPLASH_4_MAG']

zphoto, mass, SSFR, age = fdata['PHOTOZ'], fdata['MASS_MED'], fdata['SSFR_MED'], fdata['AGE']

u_B, B_V, V_r, r_ip, ip_zpp, zpp_yHSC, yHSC_Y, Y_J, J_H, H_Ks = u-B, B-V, V-r, r-ip, ip-zpp, zpp-yHSC, yHSC-Y, Y-J, J-H, H-Ks

X = np.array([u, u_B, B_V, V_r, r_ip, ip_zpp, zpp_yHSC, yHSC_Y, Y_J, J_H, H_Ks])#, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
X = X.transpose() #This makes it so that each array entry is all band magnitudes of a galaxy

#Remove problematic galaxies
good_indices = np.argwhere((np.abs(zphoto)<90) & (np.abs(mass)<90) & (np.abs(SSFR)<90) & (age>100)).flatten() #& (np.abs(X[:,-1])<=90) & (np.abs(X[:,-2])<=90)).flatten()
X, zphoto, mass, SSFR, age= X[good_indices], zphoto[good_indices], mass[good_indices], SSFR[good_indices], age[good_indices]

#Shuffle data and build training and test set (75% and 25%)
X_train, X_test, zphoto_train, zphoto_test, mass_train, mass_test, SSFR_train, SSFR_test, age_train, age_test = train_test_split(X, zphoto, mass, SSFR, age)

#Build an instance of the UMAP algorithm class and use it on the dataset
trans = umap.UMAP().fit(X_train)
train_embedding = trans.embedding_
test_embedding = trans.transform(X_test)

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
