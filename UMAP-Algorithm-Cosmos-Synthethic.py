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

X = np.array([u, g, r, i, Z, Y, J, H, Ks])
#X = np.array([H, J, Ks, Y, Z, g, i, r, u])
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
"""

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
#X_perm_noerr = X_noerr[permuted_indices]
X_perm = X[permuted_indices]
#galaxy_type_perm = galaxy_type[permuted_indices]
#z_redshift_perm = z_redshift[permuted_indices]
zspec_perm = zspec[permuted_indices]

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
CSa = axs.scatter(embedding[:, 0], embedding[:, 1], 10, c=X_perm[:,8], cmap='summer', norm=matplotlib.colors.LogNorm())
cbara = fig.colorbar(CSa)
#CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 10, c=X_perm[3][split_b], cmap='autumn_r')#, norm=matplotlib.colors.LogNorm(vmin=split,vmax=np.max(zspec_perm)))
#cbarb = fig.colorbar(CSb)
cbara.set_label('Ks-mag')
#cbarb.set_label('1.3<=Z')
plt.show()

"""
#Galaxy type map
unique_g_types = np.unique(galaxy_type_perm)

def galaxy_type_map(galaxy_type):
    for i in range(len(unique_g_types)):
        if galaxy_type == unique_g_types[i]:
            return i

galaxy_type_numbers = [galaxy_type_map(galaxy) for galaxy in galaxy_type_perm]

#visualize the distribution of galaxy types in the compressed feature space
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]})

cmap = plt.cm.jet# define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0, 8, 9)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

CS = axs[0].scatter(embedding[:, 0], embedding[:, 1], 10, c=galaxy_type_numbers, cmap=cmap)

#axs2 = fig.add_subplot()
cbar = mpl.colorbar.ColorbarBase(axs[1], cmap=cmap, norm=norm, spacing='proportional', ticks=bounds+0.5, boundaries=bounds, format='%1i')
cbar.set_ticklabels(unique_g_types)
plt.show()
"""
