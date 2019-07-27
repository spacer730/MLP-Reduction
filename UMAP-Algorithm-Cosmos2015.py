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

X = np.array([u, B, V, r, ip, zpp, yHSC, Y, J, H, Ks, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
X = X.transpose() #This makes it so that each array entry is all band magnitudes of a galaxy

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm = X[permuted_indices]
zphoto_perm = zphoto[permuted_indices]
mass_perm = mass[permuted_indices]
SSFR_perm = SSFR[permuted_indices]
age_perm = age[permuted_indices]

good_indices = np.argwhere((np.abs(zphoto_perm)<90) & (np.abs(mass_perm)<90) & (np.abs(SSFR_perm)<90) & (age_perm>100) & (np.abs(X_perm[:,-1])<=90) & (np.abs(X_perm[:,-2])<=90)).flatten()

X_perm = X_perm[good_indices]
zphoto_perm = zphoto_perm[good_indices]
mass_perm = mass_perm[good_indices]
SSFR_perm = SSFR_perm[good_indices]
age_perm = age_perm[good_indices]

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_perm)

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#Split the dataset into two different groups
split = np.mean(SSFR_perm)-np.var(SSFR_perm)**0.5
split_a = SSFR_perm<split
split_b = SSFR_perm>=split

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
#CSa = axs.scatter(embedding[:, 0][split_a], embedding[:, 1][split_a], 3, c=SSFR_perm[split_a], cmap='summer')#, norm=matplotlib.colors.LogNorm())
#cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 3, c=SSFR_perm[split_b], cmap='autumn_r')#, norm=matplotlib.colors.LogNorm())
cbarb = fig.colorbar(CSb)
#cbara.set_label('SSFR<-9.87')
cbarb.set_label('-9.87<=SSFR')
plt.show()
