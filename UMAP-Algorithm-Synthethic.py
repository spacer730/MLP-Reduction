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
sdata = Table.read('Data/Synthethic-galaxy-data-100000-samples.dat', format='ascii')
sdata = np.array(sdata)

u, g, r, i, Z = sdata['u'], sdata['g'], sdata['r'], sdata['i'], sdata['z']
u_noerr, g_noerr, r_noerr, i_noerr, Z_noerr = sdata['u_noerr'], sdata['g_noerr'], sdata['r_noerr'], sdata['i_noerr'], sdata['z_noerr']
z_redshift = sdata['z_redshift']
galaxy_type = sdata['type']

u_g, g_r, r_i, i_Z = u-g, g-r, r-i, i-Z
u_g_noerr, g_r_noerr, r_i_noerr, i_Z_noerr = u_noerr-g_noerr, g_noerr-r_noerr, r_noerr-i_noerr, i_noerr-Z_noerr

u_i, g_i, Z_i = u-i, g-i, Z-i
u_i_noerr, g_i_noerr, Z_i_noerr = u_noerr-i_noerr, g_noerr-i_noerr, Z_noerr-i_noerr

#X = np.array([u, g, r, i, Z])
#X = np.array([i, u_g, g_r, r_i, i_Z])
#X = np.array([u_g, g_r, r_i, i_Z])
X = np.array([i, u_i, g_i, r_i, Z_i])
X = X.transpose()

#X_noerr = np.array([u_noerr, g_noerr, r_noerr, i_noerr, z_noerr])
#X_noerr = np.array([i_noerr, u_g_noerr, g_r_noerr, r_i_noerr, i_Z_noerr])
#X_noerr = np.array([u_g_noerr, g_r_noerr, r_i_noerr, i_Z_noerr])
X_noerr = np.array([i_noerr, u_i_noerr, g_i_noerr, r_i_noerr, Z_i_noerr])
X_noerr = X_noerr.transpose()

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm_noerr = X_noerr[permuted_indices]
X_perm = X[permuted_indices]
galaxy_type_perm = galaxy_type[permuted_indices]
z_redshift_perm = z_redshift[permuted_indices]

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_perm_noerr)

#Split the dataset into two different groups
split = 1.3#np.mean(z_redshift_perm)+np.var(z_redshift_perm)**0.5
split_a = z_redshift_perm<split
split_b = z_redshift_perm>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(embedding[:, 0][split_a], embedding[:, 1][split_a], 3, c=z_redshift_perm[split_a], cmap='summer', norm=matplotlib.colors.LogNorm(vmin=0.01,vmax=1.3))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 3, c=z_redshift_perm[split_b], cmap='autumn_r', norm=matplotlib.colors.Normalize(vmin=split,vmax=np.max(z_redshift_perm)))#, norm=matplotlib.colors.LogNorm(vmin=split,vmax=np.max(zspec_perm)))
cbarb = fig.colorbar(CSb)
cbara.set_label('z<1.3')
cbarb.set_label('1.3<=z')
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
