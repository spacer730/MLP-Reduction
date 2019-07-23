import tensorflow as tf
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

X = np.array([u, B, V, r, ip, zpp, yHSC, Y, J, H, Ks])#, ch1, ch2, ch3, ch4]) #ch3 has many errors
X = X.transpose() #This ensures that each array entry is the 9 magnitudes of a galaxy

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm = X[permuted_indices]
zphoto_perm = zphoto[permuted_indices]
mass_perm = mass[permuted_indices]
SSFR_perm = SSFR[permuted_indices]
age_perm = age[permuted_indices]

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_perm)

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()
#Split the dataset into two different groups

"""
split = np.mean(zspec_perm)+np.var(zspec_perm)**0.5
split_a = zspec_perm<split
split_b = zspec_perm>=split

split = np.mean(z_redshift_perm)+np.var(z_redshift_perm)**0.5
split_a = z_redshift_perm<split
split_b = z_redshift_perm>=split
"""

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(embedding[:, 0], embedding[:, 1], 10)#, c=zphto_perm, cmap='summer', norm=matplotlib.colors.LogNorm())
#cbara = fig.colorbar(CSa)
#CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 10, c=X_perm[3][split_b], cmap='autumn_r')#, norm=matplotlib.colors.LogNorm(vmin=split,vmax=np.max(zspec_perm)))
#cbarb = fig.colorbar(CSb)
#cbara.set_label('zphoto')
#cbarb.set_label('1.3<=Z')
plt.show()
