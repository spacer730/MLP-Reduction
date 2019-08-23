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

#Load the data and reshape it in the form we want
print('Openining the data files')
fdata_1 = fits.open('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_no_doubles_G10COSMOSCatv05-COSMOS2015.fits')[1].data
fdata_2 = fits.open('Data/G10COSMOSCatv05_Z_USE_1_2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_no_doubles_C3R2.fits')[1].data
fdata_3 = fits.open('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_doubles_G10COSMOSCatv05-COSMOS2015.fits')[1].data
print('File is open')

u, B, V, r, ip, zpp = fdata_1['u_MAG_AUTO'], fdata_1['B_MAG_AUTO'], fdata_1['V_MAG_AUTO'], fdata_1['r_MAG_AUTO'], fdata_1['ip_MAG_AUTO'], fdata_1['zpp_MAG_AUTO']
Y, J, H, Ks = fdata_1['Y_MAG_AUTO'], fdata_1['J_MAG_AUTO'], fdata_1['H_MAG_AUTO'], fdata_1['Ks_MAG_AUTO']
ch1, ch2, ch3, ch4 = fdata_1['SPLASH_1_MAG'], fdata_1['SPLASH_2_MAG'], fdata_1['SPLASH_3_MAG'], fdata_1['SPLASH_4_MAG']

zphoto, mass, SSFR, age = fdata_1['PHOTOZ'], fdata_1['MASS_MED'], fdata_1['SSFR_MED'], fdata_1['AGE']
source_type = fdata_1['type']

Z_C3R2 = fdata_1['Redshift']

u, B, V = np.append(u,fdata_2['u_MAG_AUTO']), np.append(B,fdata_2['B_MAG_AUTO']), np.append(V,fdata_2['V_MAG_AUTO'])
r, ip, zpp = np.append(r,fdata_2['r_MAG_AUTO']), np.append(ip,fdata_2['ip_MAG_AUTO']), np.append(zpp,fdata_2['zpp_MAG_AUTO'])
Y, J, H, Ks = np.append(Y,fdata_2['Y_MAG_AUTO']), np.append(J,fdata_2['J_MAG_AUTO']), np.append(H,fdata_2['H_MAG_AUTO']), np.append(Ks,fdata_2['Ks_MAG_AUTO'])
ch1, ch2, ch3, ch4 = np.append(ch1,fdata_2['SPLASH_1_MAG']), np.append(ch2,fdata_2['SPLASH_2_MAG']), np.append(ch3,fdata_2['SPLASH_3_MAG']), np.append(ch4,fdata_2['SPLASH_4_MAG'])

zphoto, mass, SSFR, age = np.append(zphoto,fdata_2['PHOTOZ']), np.append(mass,fdata_2['MASS_MED']), np.append(SSFR,fdata_2['SSFR_MED']), np.append(age,fdata_2['AGE'])
source_type = np.append(source_type,fdata_2['type'])

Z_G10COSMOS = fdata_2['Z_BEST']
Z_GEN = fdata_2['Z_GEN']
Z_USE = fdata_2['Z_USE']

u, B, V = np.append(u,fdata_3['u_MAG_AUTO_1']), np.append(B,fdata_3['B_MAG_AUTO_1']), np.append(V,fdata_3['V_MAG_AUTO_1'])
r, ip, zpp = np.append(r,fdata_3['r_MAG_AUTO_1']), np.append(ip,fdata_3['ip_MAG_AUTO_1']), np.append(zpp,fdata_3['zpp_MAG_AUTO_1'])
Y, J, H, Ks = np.append(Y,fdata_3['Y_MAG_AUTO_1']), np.append(J,fdata_3['J_MAG_AUTO_1']), np.append(H,fdata_3['H_MAG_AUTO_1']), np.append(Ks,fdata_3['Ks_MAG_AUTO_1'])
ch1, ch2, ch3, ch4 = np.append(ch1,fdata_3['SPLASH_1_MAG_1']), np.append(ch2,fdata_3['SPLASH_2_MAG_1']), np.append(ch3,fdata_3['SPLASH_3_MAG_1']), np.append(ch4,fdata_3['SPLASH_4_MAG_1'])

zphoto, mass, SSFR, age = np.append(zphoto,fdata_3['PHOTOZ_1']), np.append(mass,fdata_3['MASS_MED_1']), np.append(SSFR,fdata_3['SSFR_MED_1']), np.append(age,fdata_3['AGE_1'])
source_type = np.append(source_type,fdata_3['type_1'])

Z_C3R2_doubles = fdata_3['Redshift']
Z_G10COSMOS_doubles = fdata_3['Z_BEST']
Z_GEN_doubles = fdata_3['Z_GEN']
Z_USE_doubles = fdata_3['Z_USE']

Z_spec = np.append(Z_C3R2, Z_G10COSMOS)
Z_spec = np.append(Z_spec, Z_C3R2_doubles)

#Some interesting lines to analyze the data
np.abs(Z_G10COSMOS_doubles-Z_C3R2_doubles)>0.1
source_type[source_type!=0]

u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks = u-B, B-V, V-r, r-ip, ip-zpp, zpp-Y, Y-J, J-H, H-Ks

X = np.array([u, u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks])#, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
X = X.transpose() #This ensures that each array entry are the colors/magnitudes of a galaxy

"""
def change_numbers(in_c):
    out_c = np.zeros(len(in_c))
    for i in range(len(in_c)):
        if in_c[i] == -9:
            out_c[i] = -8
        elif in_c[i] == 2:
            out_c[i] = -4
        else:
            out_c[i] = in_c[i]
    return out_c

def change_numbers_2(in_c):
    out_c = np.zeros(len(in_c))
    for i in range(len(in_c)):
        if in_c[i] == -9:
            out_c[i] = 3
        else:
            out_c[i] = in_c[i]
    return out_c

new_source_type = change_numbers(source_type)
new_source_type_2 = change_numbers_2(source_type)
"""

"""
#Remove problematic galaxies
#It turns out that 16% of galaxies with Z_BEST > 1.4 are bad galaxies and only 0.6% of galaxies with Z_BEST <= 1.4 are considered bad galaxies
good_indices = np.argwhere((np.abs(mass)<90) & (np.abs(SSFR)<90) & (age>100)).flatten()
bad_indices = np.argwhere((np.abs(mass)>=90) | (np.abs(SSFR)>=90) | (age<=100)).flatten()

X, mass, SSFR, age = X[good_indices], mass[good_indices], SSFR[good_indices], age[good_indices]

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm = X[permuted_indices]
Z_BEST_perm = Z_BEST[permuted_indices]
Z_GEN_perm = Z_GEN[permuted_indices]
Z_USE_perm = Z_USE[permuted_indices]

mass_perm = mass[permuted_indices]
SSFR_perm = SSFR[permuted_indices]
age_perm = age[permuted_indices]
"""

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X)

#Selecting the outliers
def outlier(x, y, eps, zspec_bound, minimum=True):
    if minimum:
        return np.argwhere((np.abs(embedding[:,0]-x)<eps) & (np.abs(embedding[:,1]-y)<eps) & (Z_spec>zspec_bound)).flatten()
    else:
        return np.argwhere((np.abs(embedding[:,0]-x)<eps) & (np.abs(embedding[:,1]-y)<eps) & (Z_spec<=zspec_bound)).flatten()
    
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


#Split the dataset into two different groups
split = 0.93#np.mean(Z_spec)+np.var(Z_spec)**0.5
split_a = Z_spec<split
split_b = Z_spec>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

"""
for i in range(len(outliers)):
    axs.text(embedding[:,0][outliers[i]], embedding[:,1][outliers[i]], str(i+1), color = 'k')
"""

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(embedding[:, 0][(source_type == 0) & (Z_spec<0.93)], embedding[:, 1][(source_type == 0) & (Z_spec<0.93)], s=1, c=Z_spec[(source_type == 0) & (Z_spec<0.93)], cmap='summer', norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=split))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][(source_type == 0) & (Z_spec>=0.93)], embedding[:, 1][(source_type == 0) & (Z_spec>=0.93)], 1, c=Z_spec[(source_type == 0) & (Z_spec>=0.93)], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(Z_spec)))
cbarb = fig.colorbar(CSb)
axs.text(5,8, 'Source_type == 0')
cbara.set_label('Z_spec < 0.93')
cbarb.set_label('0.93 <= Z_spec')
axs.set_xlim([-9.75,8.8])
axs.set_ylim([-10.4,8.7])
plt.show()

"""
#Object type map
fig, axs = plt.subplots()
custom_cmap = plt.cm.get_cmap('jet', 4)#

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(embedding[:, 0], embedding[:, 1], s=[new_source_type_2[i]+0.1 for i in range(len(new_source_type_2))], c=new_source_type_2, cmap=custom_cmap)#, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=split))
cbara = fig.colorbar(CSa)
#CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 1, c=Z_spec[split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(Z_spec)))
#cbarb = fig.colorbar(CSb)
cbara.set_label('Type')
cbara.set_ticks([0.375, 1.5-0.375, 1.5+0.375, 1.5+3*0.375])
cbara.set_ticklabels(['Galaxies', 'Stars', 'X-ray sources', 'No-fit'])
#cbarb.set_label('0.93 <= Z_spec')
plt.show()
"""
