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
Y, J, H, Ks = fdata['Y_MAG_AUTO'], fdata['J_MAG_AUTO'], fdata['H_MAG_AUTO'], fdata['Ks_MAG_AUTO']
ch1, ch2, ch3, ch4 = fdata['SPLASH_1_MAG'], fdata['SPLASH_2_MAG'], fdata['SPLASH_3_MAG'], fdata['SPLASH_4_MAG']

zphoto, mass, SSFR, age = fdata['PHOTOZ'], fdata['MASS_MED'], fdata['SSFR_MED'], fdata['AGE']
source_type = fdata['type']

u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks = u-B, B-V, V-r, r-ip, ip-zpp, zpp-Y, Y-J, J-H, H-Ks

X = np.array([u, u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks])#, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
X = X.transpose() #This makes it so that each array entry is all band magnitudes of a galaxy

#Select only object type galaxy and remove problematic galaxies
good_indices = np.argwhere((np.abs(zphoto)<90) & (np.abs(mass)<90) & (np.abs(SSFR)<90) & (age>100) & (source_type == 0)).flatten() #& (np.abs(X[:,-1])<=90) & (np.abs(X[:,-2])<=90)).flatten()
X, zphoto, mass, SSFR, age= X[good_indices], zphoto[good_indices], mass[good_indices], SSFR[good_indices], age[good_indices]

#Shuffle data and build training and test set (75% and 25%)
X_train, X_test, zphoto_train, zphoto_test, mass_train, mass_test, SSFR_train, SSFR_test, age_train, age_test = train_test_split(X, zphoto, mass, SSFR, age)

#Build an instance of the UMAP algorithm class and use it on the dataset
trans = umap.UMAP().fit(X_train)
train_embedding = trans.embedding_
test_embedding = trans.transform(X_test)

#Take data from bin out of UMAP
binned_indices_train = np.argwhere((zphoto_train>=0.7) & (zphoto_train<0.9)).flatten()
X_train_bin = X_train[binned_indices_train]
train_embedding_bin = train_embedding[binned_indices_train]
zphoto_train_bin, mass_train_bin, SSFR_train_bin, age_train_bin = zphoto_train[binned_indices_train], mass_train[binned_indices_train], SSFR_train[binned_indices_train], age_train[binned_indices_train]

binned_indices_test = np.argwhere((zphoto_test>=0.7) & (zphoto_test<0.9)).flatten()
X_test_bin = X_test[binned_indices_test]
test_embedding_bin = test_embedding[binned_indices_test]
zphoto_test_bin, mass_test_bin, SSFR_test_bin, age_test_bin = zphoto_test[binned_indices_test], mass_test[binned_indices_test], SSFR_test[binned_indices_test], age_test[binned_indices_test]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#Training knn regressor for redshift on both the embedded data as well as the original data
neigh = [KNeighborsRegressor(n_neighbors=i, weights='distance').fit(train_embedding_bin, mass_train_bin) for i in range(1,100)]
mass_test_pred = [neigh[i].predict(test_embedding_bin) for i in range(len(neigh))]
score_pred = [neigh[i].score(test_embedding_bin, mass_test_bin) for i in range(len(neigh))]
rmse_pred = [rmse(mass_test_pred[i], mass_test_bin) for i in range(len(mass_test_pred))]

neigh2 = [KNeighborsRegressor(n_neighbors=i, weights='distance').fit(X_train_bin, mass_train_bin) for i in range(1,15)]
mass_test_pred_2 = [neigh2[i].predict(X_test_bin) for i in range(len(neigh2))]
score_pred_2 = [neigh2[i].score(X_test_bin, mass_test_bin) for i in range(len(neigh2))]
rmse_pred_2 = [rmse(mass_test_pred_2[i], mass_test_bin) for i in range(len(mass_test_pred_2))]

"""
plt.plot([i+1 for i in range(len(rmse_pred_2))], rmse_pred_2)
plt.xlabel('# nearest neighbors')
plt.ylabel('RMSE')
plt.text(12, 0.21, 'mass')
plt.show()
"""

#Training support vector regressor for redshift on both the embedded data as well as the original data
svr1 = SVR().fit(train_embedding_bin, mass_train_bin)
mass_test_pred_3 = svr1.predict(test_embedding_bin)
score_pred_3 = svr1.score(test_embedding_bin, mass_test_bin)
rmse_pred_3 = rmse(mass_test_pred_3, mass_test_bin)

svr2 = SVR().fit(X_train_bin, mass_train_bin)
mass_test_pred_4 = svr2.predict(X_test_bin)
score_pred_4 = svr2.score(X_test_bin, mass_test_bin)
rmse_pred_4 = rmse(mass_test_pred_4, mass_test_bin)

#Split the dataset into two different groups
split = 9#np.mean(age_train)+np.var(age_train)**0.5
split_a = mass_test_pred_2[9]<split
split_b = mass_test_pred_2[9]>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
#CSa = axs.scatter(test_embedding_bin[:, 0][split_a], test_embedding_bin[:, 1][split_a], 3, c=mass_test_pred_2[9][split_a], cmap='summer', norm=matplotlib.colors.Normalize(vmin=7,vmax=9))
#cbara = fig.colorbar(CSa)
axs.text(5,6,'0.7<=z<0.9',color='k')
CSb = axs.scatter(test_embedding_bin[:, 0][split_b], test_embedding_bin[:, 1][split_b], 3, c=mass_test_pred_2[9][split_b], cmap='autumn_r', norm=matplotlib.colors.Normalize(vmin=9,vmax=12))
cbarb = fig.colorbar(CSb)
#cbara.set_label('mass<9')
cbarb.set_label('9<=mass')
axs.set_xlim([-6.5,10.25])
axs.set_ylim([-10.5,10.5])
plt.show()
