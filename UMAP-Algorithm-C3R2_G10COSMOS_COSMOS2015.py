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
fdata = fits.open('Data/KOA_c3r2_v2_G10COSMOSCatv05_Z_USE_1_2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90.fits')[1].data
print('File is open')

u_1, B_1, V_1, r_1, ip_1, zpp_1 = fdata['u_MAG_AUTO_1'], fdata['B_MAG_AUTO_1'], fdata['V_MAG_AUTO_1'], fdata['r_MAG_AUTO_1'], fdata['ip_MAG_AUTO_1'], fdata['zpp_MAG_AUTO_1']
Y_1, J_1, H_1, Ks_1 = fdata['Y_MAG_AUTO_1'], fdata['J_MAG_AUTO_1'], fdata['H_MAG_AUTO_1'], fdata['Ks_MAG_AUTO_1']
ch1_1, ch2_1, ch3_1, ch4_1 = fdata['SPLASH_1_MAG_1'], fdata['SPLASH_2_MAG_1'], fdata['SPLASH_3_MAG_1'], fdata['SPLASH_4_MAG_1']

zphoto_1, mass_1, SSFR_1, age_1 = fdata['PHOTOZ_1'], fdata['MASS_MED_1'], fdata['SSFR_MED_1'], fdata['AGE_1']
source_type_1 = fdata['type_1']

Z_C3R2 = fdata['Redshift']

u_2, B_2, V_2, r_2, ip_2, zpp_2 = fdata['u_MAG_AUTO_2'], fdata['B_MAG_AUTO_2'], fdata['V_MAG_AUTO_2'], fdata['r_MAG_AUTO_2'], fdata['ip_MAG_AUTO_2'], fdata['zpp_MAG_AUTO_2']
Y_2, J_2, H_2, Ks_2 = fdata['Y_MAG_AUTO_2'], fdata['J_MAG_AUTO_2'], fdata['H_MAG_AUTO_2'], fdata['Ks_MAG_AUTO_2']
ch1_2, ch2_2, ch3_2, ch4_2 = fdata['SPLASH_2_MAG_2'], fdata['SPLASH_2_MAG_2'], fdata['SPLASH_3_MAG_2'], fdata['SPLASH_4_MAG_2']

zphoto_2, mass_2, SSFR_2, age_2 = fdata['PHOTOZ_2'], fdata['MASS_MED_2'], fdata['SSFR_MED_2'], fdata['AGE_2']
source_type_2 = fdata['type_2']

Z_G10Cosmos = fdata['Z_BEST']
Z_GEN = fdata['Z_GEN']
Z_USE = fdata['Z_USE']

u_B_1, B_V_1, V_r_1, r_ip_1, ip_zpp_1, zpp_Y_1, Y_J_1, J_H_1, H_Ks_1 = u_1-B_1, B_1-V_1, V_1-r_1, r_1-ip_1, ip_1-zpp_1, zpp_1-Y_1, Y_1-J_1, J_1-H_1, H_1-Ks_1
u_B_2, B_V_2, V_r_2, r_ip_2, ip_zpp_2, zpp_Y_2, Y_J_2, J_H_2, H_Ks_2 = u_2-B_2, B_2-V_2, V_2-r_2, r_2-ip_2, ip_2-zpp_2, zpp_2-Y_2, Y_2-J_2, J_2-H_2, H_2-Ks_2

#Continue weaving these two different data (_1 and _2) into one thing

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
