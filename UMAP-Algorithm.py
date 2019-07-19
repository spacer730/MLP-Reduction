import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
from astropy.io import fits
from astropy.table import Table

np.random.seed(33)

#Load the data en reshape it in the form we want
"""
print('Openining the data file')
fdata = fits.open('Data/COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_removed_magnitudes_larger_90.fits')[1].data
print('File is open')
Ha = fdata['MAG_GAAPadapt_H']
H = fdata['MAG_GAAP_H']
Hflux = fdata['FLUX_GAAP_H']
Ja = fdata['MAG_GAAPadapt_J']
J = fdata['MAG_GAAP_J']
Jflux = fdata['FLUX_GAAP_J']
Ksa = fdata['MAG_GAAPadapt_Ks']
Ks = fdata['MAG_GAAP_Ks']
Ksflux = fdata['FLUX_GAAP_Ks']
Ya = fdata['MAG_GAAPadapt_Y']
Y = fdata['MAG_GAAP_Y']
Yflux = fdata['FLUX_GAAP_Y']
Za = fdata['MAG_GAAPadapt_Z']
Z = fdata['MAG_GAAP_Z']
Zflux = fdata['FLUX_GAAP_Z']
ga = fdata['MAG_GAAPadapt_g']
g = fdata['MAG_GAAP_g']
gflux = fdata['FLUX_GAAP_g']
ia = fdata['MAG_GAAPadapt_i']
i = fdata['MAG_GAAP_i']
iflux = fdata['FLUX_GAAP_i']
ra = fdata['MAG_GAAPadapt_r']
r = fdata['MAG_GAAP_r']
rflux = fdata['FLUX_GAAP_r']
ua = fdata['MAG_GAAPadapt_u']
u = fdata['MAG_GAAP_u']
uflux = fdata['FLUX_GAAP_u']
zspec = fdata['zspec']

X = np.array([H, J, Ks, Y, Z, g, i, r, u])
X = X.transpose() #This ensures that each array entry is the 9 magnitudes of a galaxy
"""

sdata = Table.read('Data/Synthethic-galaxy-data-20000-samples.dat', format='ascii')
sdata = np.array(sdata)

u = sdata['u']
g = sdata['g']
r = sdata['r']
i = sdata['i']
z = sdata['z']
u_noerr = sdata['u_noerr']
g_noerr = sdata['g_noerr']
r_noerr = sdata['r_noerr']
i_noerr = sdata['i_noerr']
z_noerr = sdata['z_noerr']
z_redshift = sdata['z_redshift']
galaxy_type = sdata['type']

X = np.array([u, g, r, i, z])
X = X.transpose()

X_noerr = np.array([u_noerr, g_noerr, r_noerr, i_noerr, z_noerr])
X_noerr = X_noerr.transpose()

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm_noerr = X_noerr[permuted_indices]
X_perm = X[permuted_indices]
z_redshift_perm = z_redshift[permuted_indices]
#zspec_perm = zspec[permuted_indices]

cut_off = int(0.8*len(X_perm))
X_train = X_perm[0:cut_off]
X_test = X_perm[cut_off::]

#Normalize data
X_train = (X_train-np.min(X))/(np.max(X)-np.min(X))
X_test = (X_test-np.min(X))/(np.max(X)-np.min(X))

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_perm)

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()
CS = axs.scatter(embedding[:, 0], embedding[:, 1], 10, c=z_redshift_perm, cmap='Blues') #x,y coordinates and the size of the dot
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Redshift')
plt.show()
