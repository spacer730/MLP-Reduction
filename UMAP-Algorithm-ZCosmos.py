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

np.random.seed(33)

#Load the data en reshape it in the form we want
print('Openining the data file')
fdata = Table.read("Data/COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_removed_magnitudes_larger_90.fits",format='fits')
#fdata = fits.open('Data/COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_removed_magnitudes_larger_90.fits')[1].data
print('File is open')
u = fdata['MAG_GAAP_u']
g = fdata['MAG_GAAP_g']
r = fdata['MAG_GAAP_r']
i = fdata['MAG_GAAP_i']
Z = fdata['MAG_GAAP_Z']
Y = fdata['MAG_GAAP_Y']
J = fdata['MAG_GAAP_J']
H = fdata['MAG_GAAP_H']
Ks = fdata['MAG_GAAP_Ks']
zspec = fdata['zspec']

u_g = u-g
g_r = g-r
r_i = r-i
i_Z = i-Z
Z_Y = Z-Y
Y_J = Y-J
J_H = J-H
H_Ks = H-Ks

#X = np.array([u, g, r, i, Z, Y, J, H, Ks])
X = np.array([i, u_g, g_r, r_i, i_Z, Z_Y, Y_J, J_H, H_Ks])
X = X.transpose() #This ensures that each array entry are the colors/magnitudes of a galaxy

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm = X[permuted_indices]
zspec_perm = zspec[permuted_indices]

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_perm)

#Selecting the outliers
def outlier(x, y, eps, zspec_min):
    return np.argwhere((np.abs(embedding[:,0]-x)<eps) & (np.abs(embedding[:,1]-y)<eps) & (zspec_perm>zspec_min)).flatten()

#Select the indices of the 'normal' data around the outlier
def orbiter(outlier_index, eps, zspec_max):
    return np.argwhere((np.abs(embedding[:,0]-embedding[:,0][outliers[outlier_index]])<eps) & (np.abs(embedding[:,1]-embedding[:,1][outliers[outlier_index]])<eps) & (zspec_perm<zspec_max)).flatten()

outliers = np.array([outlier(7.2,-2.0,0.2,2)[0], outlier(3.3,-7.3,0.2,1.5)[0], outlier(-6.3,1.3,0.2,2.5)[0], outlier(2.2,-0.4,0.2,2.5)[0], outlier(9.2,-0.4,0.2,1.5)[0], outlier(6.14,-7.01,0.2,2)[0], outlier(2.55,-1.23,0.2,2)[0], outlier(-3.47,0.37,0.2,2)[0]])
normal_orbiters = np.array([orbiter(0,0.07,split), orbiter(1,0.07,split), orbiter(2,0.1,split), orbiter(3,0.05,split), orbiter(4,0.0521,split), orbiter(5,0.084,split), orbiter(6,0.1,split), orbiter(7,0.1,split)])
normal_orbiters = normal_orbiters.flatten()

og_normal_orbiters = permuted_indices[normal_orbiters]
data_normal_orbiters = fdata[og_normal_orbiters]
data_normal_orbiters.write("COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_removed_magnitudes_larger_90_normal_orbiters.fits",format='fits')

#"""
#Changing the indices from the permuted array to the original array and then selecting these from the original data and writing to a file
og_outliers = permuted_indices[outliers]
data_outliers = fdata[og_outliers]
data_outliers.write("COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_removed_magnitudes_larger_90_outliers.fits",format='fits')
#"""

#Split the dataset into two different groups
split = np.mean(zspec_perm)+np.var(zspec_perm)**0.5
split_a = zspec_perm<split
split_b = zspec_perm>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

for i in range(len(outliers)):
    axs.text(embedding[:,0][outliers[i]], embedding[:,1][outliers[i]], str(i+1), color = 'k')

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(embedding[:, 0][normal_orbiters], embedding[:, 1][normal_orbiters], 5, c=zspec_perm[normal_orbiters], cmap='summer', norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=split))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 5, c=zspec_perm[split_b], cmap='autumn_r')#, norm=matplotlib.colors.LogNorm(vmin=split,vmax=np.max(zspec_perm)))
cbarb = fig.colorbar(CSb)
cbara.set_label('zspec < 1.242')
cbarb.set_label('1.242 <= Z')
plt.show()
