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
fdata = fits.open('Data/G10COSMOSCatv05_Z_USE_1_2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90.fits')[1].data
print('File is open')
"""
U = fdata['U_08']
B = fdata['B_08']
V = fdata['V_08']
G = fdata['G_08']
R = fdata['R_08']
I = fdata['I_08']
"""

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

Z_BEST = fdata['Z_BEST']
Z_GEN = fdata['Z_GEN']
Z_USE = fdata['Z_USE']

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

#X = np.array([u, g, r, i, Z, Y, J, H, Ks])
X = np.array([u, u_B, B_V, V_r, r_ip, ip_zpp, zpp_yHSC, yHSC_Y, Y_J, J_H, H_Ks])#, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
X = X.transpose() #This ensures that each array entry are the colors/magnitudes of a galaxy

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm = X[permuted_indices]
Z_BEST_perm = Z_BEST[permuted_indices]
Z_GEN_perm = Z_GEN[permuted_indices]
Z_USE_perm = Z_USE[permuted_indices]

mass_perm = mass[permuted_indices]
SSFR_perm = SSFR[permuted_indices]
age_perm = age[permuted_indices]

#Remove problematic galaxies
#It turns out that 16% of galaxies with Z_BEST > 1.4 are bad galaxies and only 0.6% of galaxies with Z_BEST <= 1.4 are considered bad galaxies
good_indices = np.argwhere((np.abs(mass_perm)<90) & (np.abs(SSFR_perm)<90) & (age_perm>100)).flatten()
bad_indices = np.argwhere((np.abs(mass_perm)>=90) | (np.abs(SSFR_perm)>=90) | (age_perm<=100)).flatten()

X_perm = X_perm[good_indices]
Z_BEST_perm = Z_BEST_perm[good_indices]
Z_GEN_perm = Z_GEN[good_indices]
Z_USE_perm = Z_USE[good_indices]
mass_perm = mass_perm[good_indices]
SSFR_perm = SSFR_perm[good_indices]
age_perm = age_perm[good_indices]

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_perm)

"""
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

#Changing the indices from the permuted array to the original array and then selecting these from the original data and writing to a file
#og_outliers = permuted_indices[outliers]
#data_outliers = fdata[og_outliers]
#data_outliers.write("COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_removed_magnitudes_larger_90_outliers.fits",format='fits')
"""

#Split the dataset into two different groups
split = -10#np.mean(SSFR_perm)-np.var(SSFR_perm)**0.5
split_a = SSFR_perm<split
split_b = SSFR_perm>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

"""
for i in range(len(outliers)):
    axs.text(embedding[:,0][outliers[i]], embedding[:,1][outliers[i]], str(i+1), color = 'k')
"""

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
#CSa = axs.scatter(embedding[:, 0][split_a], embedding[:, 1][split_a], 5, c=SSFR_perm[split_a], cmap='summer')#, norm=matplotlib.colors.LogNorm())
#cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 5, c=SSFR_perm[split_b], cmap='autumn_r')#, norm=matplotlib.colors.LogNorm())
cbarb = fig.colorbar(CSb)
#cbara.set_label('SSFR < -10')
cbarb.set_label('-10 <= SSFR')
plt.show()
