import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
import matplotlib.colors
import matplotlib as mpl
from astropy.io import fits
from astropy.table import Table
from numpy.ma import masked_array

np.random.seed(33)

#Load the data en reshape it in the form we want
print('Openining the data file')
fdata = fits.open('Data/COSMOS2015_Laigle+_v1.1_removed_optical_nir_magnitudes_larger_90.fits')[1].data
print('File is open')

u = fdata['u_MAG_AUTO']
B = fdata['B_MAG_AUTO']
V = fdata['V_MAG_AUTO']
r = fdata['r_MAG_AUTO']
ip = fdata['ip_MAG_AUTO']
zpp = fdata['zpp_MAG_AUTO']
yHSC = fdata['yHSC_MAG_AUTO']
Y = fdata['Y_MAG_AUTO']
J = fdata['J_MAG_AUTO']
H = fdata['H_MAG_AUTO']
Ks = fdata['Ks_MAG_AUTO']
ch1 = fdata['SPLASH_1_MAG']
ch2 = fdata['SPLASH_2_MAG']
ch3 = fdata['SPLASH_3_MAG']
ch4 = fdata['SPLASH_4_MAG']

zphoto = fdata['PHOTOZ']
mass = fdata['MASS_MED']
SSFR = fdata['SSFR_MED']
age = fdata['AGE']

u_B = u-B
B_V = B-V
V_r = V-r
r_ip = r-ip
ip_zpp = ip-zpp
zpp_yHSC = zpp-yHSC
yHSC_Y = yHSC-Y
Y_J = Y-J
J_H = J-H
H_Ks = H-Ks

X = np.array([u, u_B, B_V, V_r, r_ip, ip_zpp, zpp_yHSC, yHSC_Y, Y_J, J_H, H_Ks])#, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
X = X.transpose() #This makes it so that each array entry is all band magnitudes of a galaxy

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm = X[permuted_indices]
#X_perm = X_perm.reshape((np.shape(X_perm)[0],np.shape(X_perm)[-1]))
zphoto_perm = zphoto[permuted_indices]#.reshape(-1)
mass_perm = mass[permuted_indices]#.reshape(-1)
SSFR_perm = SSFR[permuted_indices]#.reshape(-1)
age_perm = age[permuted_indices]#.reshape(-1)

#Remove problematic galaxies
good_indices = np.argwhere((np.abs(zphoto_perm)<90) & (np.abs(mass_perm)<90) & (np.abs(SSFR_perm)<90) & (age_perm>100)).flatten() #& (np.abs(X_perm[:,-1])<=90) & (np.abs(X_perm[:,-2])<=90)).flatten()

X_perm = X_perm[good_indices]
zphoto_perm = zphoto_perm[good_indices]
mass_perm = mass_perm[good_indices]
SSFR_perm = SSFR_perm[good_indices]
age_perm = age_perm[good_indices]

#Take only data from a certain bin then throw this into UMAP
binned_indices = np.argwhere((zphoto_perm >= 0.9) & (zphoto_perm < 1.1)).flatten()

X_perm_bin = X_perm[binned_indices]
zphoto_perm_bin = zphoto_perm[binned_indices]
mass_perm_bin = mass_perm[binned_indices]
SSFR_perm_bin = SSFR_perm[binned_indices]
age_perm_bin = age_perm[binned_indices]

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_perm_bin)

"""
#Take data from bin out of UMAP
#binned_indices = np.argwhere((zphoto_perm >= 1.1) & (zphoto_perm<1.3)).flatten()
binned_indices = np.argwhere((X_perm[:,0] >= 23) & (X_perm[:,0] < 24)).flatten()

X_perm_bin = X_perm[binned_indices]
embedding_bin = embedding[binned_indices]

zphoto_perm_bin = zphoto_perm[binned_indices]
mass_perm_bin = mass_perm[binned_indices]
SSFR_perm_bin = SSFR_perm[binned_indices]
age_perm_bin = age_perm[binned_indices]
"""

#Split the dataset into two different groups
split = -9.25#np.mean(zphoto_perm_bin)+np.var(zphoto_perm_bin)**0.5
split_a = SSFR_perm_bin<split
split_b = SSFR_perm_bin>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
#CSa = axs.scatter(embedding[:, 0][split_a], embedding[:, 1][split_a], 3, c=SSFR_perm_bin[split_a], cmap='summer')#, norm=matplotlib.colors.LogNorm())#, norm=matplotlib.colors.Normalize(vmin=7,vmax=9))
#cbara = fig.colorbar(CSa)
axs.text(5,6,'0.9<=z<1.1',color='k')
CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 3, c=SSFR_perm_bin[split_b], cmap='autumn_r')#, norm=matplotlib.colors.LogNorm())
cbarb = fig.colorbar(CSb)
#cbara.set_label('SSFR < -9.25')
cbarb.set_label('-9.25 <= SSFR')
plt.show()
