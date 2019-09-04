import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
import matplotlib.colors
import matplotlib as mpl
import copy
from astropy.io import fits
from astropy.table import Table

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR

#Load the data and reshape it in the form we want
print('Openining the data files')
fdata_1 = fits.open('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_no_doubles_G10COSMOSCatv05-COSMOS2015.fits')[1].data
fdata_2 = fits.open('Data/G10COSMOSCatv05_Z_USE_1_2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_no_doubles_C3R2.fits')[1].data
fdata_3 = fits.open('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_doubles_G10COSMOSCatv05-COSMOS2015.fits')[1].data
#fdata_1 = Table.read('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_no_doubles_G10COSMOSCatv05-COSMOS2015.fits',format='fits')
#fdata_2 = Table.read('Data/G10COSMOSCatv05_Z_USE_1_2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_no_doubles_C3R2.fits',format='fits')
#fdata_3 = Table.read('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_doubles_G10COSMOSCatv05-COSMOS2015.fits',format='fits')

print('File is open')

u, B, V, r, ip, zpp = fdata_1['u_MAG_AUTO'], fdata_1['B_MAG_AUTO'], fdata_1['V_MAG_AUTO'], fdata_1['r_MAG_AUTO'], fdata_1['ip_MAG_AUTO'], fdata_1['zpp_MAG_AUTO']
Y, J, H, Ks = fdata_1['Y_MAG_AUTO'], fdata_1['J_MAG_AUTO'], fdata_1['H_MAG_AUTO'], fdata_1['Ks_MAG_AUTO']
ch1, ch2, ch3, ch4 = fdata_1['SPLASH_1_MAG'], fdata_1['SPLASH_2_MAG'], fdata_1['SPLASH_3_MAG'], fdata_1['SPLASH_4_MAG']

zphoto, mass, SSFR, age = fdata_1['PHOTOZ'], fdata_1['MASS_MED'], fdata_1['SSFR_MED'], fdata_1['AGE']
source_type = fdata_1['TYPE']

Z_C3R2 = fdata_1['Redshift']

u, B, V = np.append(u,fdata_2['u_MAG_AUTO']), np.append(B,fdata_2['B_MAG_AUTO']), np.append(V,fdata_2['V_MAG_AUTO'])
r, ip, zpp = np.append(r,fdata_2['r_MAG_AUTO']), np.append(ip,fdata_2['ip_MAG_AUTO']), np.append(zpp,fdata_2['zpp_MAG_AUTO'])
Y, J, H, Ks = np.append(Y,fdata_2['Y_MAG_AUTO']), np.append(J,fdata_2['J_MAG_AUTO']), np.append(H,fdata_2['H_MAG_AUTO']), np.append(Ks,fdata_2['Ks_MAG_AUTO'])
ch1, ch2, ch3, ch4 = np.append(ch1,fdata_2['SPLASH_1_MAG']), np.append(ch2,fdata_2['SPLASH_2_MAG']), np.append(ch3,fdata_2['SPLASH_3_MAG']), np.append(ch4,fdata_2['SPLASH_4_MAG'])

zphoto, mass, SSFR, age = np.append(zphoto,fdata_2['PHOTOZ']), np.append(mass,fdata_2['MASS_MED']), np.append(SSFR,fdata_2['SSFR_MED']), np.append(age,fdata_2['AGE'])
source_type = np.append(source_type,fdata_2['TYPE'])

Z_G10COSMOS = fdata_2['Z_BEST']
Z_GEN = fdata_2['Z_GEN']
Z_USE = fdata_2['Z_USE']

u, B, V = np.append(u,fdata_3['u_MAG_AUTO_1']), np.append(B,fdata_3['B_MAG_AUTO_1']), np.append(V,fdata_3['V_MAG_AUTO_1'])
r, ip, zpp = np.append(r,fdata_3['r_MAG_AUTO_1']), np.append(ip,fdata_3['ip_MAG_AUTO_1']), np.append(zpp,fdata_3['zpp_MAG_AUTO_1'])
Y, J, H, Ks = np.append(Y,fdata_3['Y_MAG_AUTO_1']), np.append(J,fdata_3['J_MAG_AUTO_1']), np.append(H,fdata_3['H_MAG_AUTO_1']), np.append(Ks,fdata_3['Ks_MAG_AUTO_1'])
ch1, ch2, ch3, ch4 = np.append(ch1,fdata_3['SPLASH_1_MAG_1']), np.append(ch2,fdata_3['SPLASH_2_MAG_1']), np.append(ch3,fdata_3['SPLASH_3_MAG_1']), np.append(ch4,fdata_3['SPLASH_4_MAG_1'])

zphoto, mass, SSFR, age = np.append(zphoto,fdata_3['PHOTOZ_1']), np.append(mass,fdata_3['MASS_MED_1']), np.append(SSFR,fdata_3['SSFR_MED_1']), np.append(age,fdata_3['AGE_1'])
source_type = np.append(source_type,fdata_3['TYPE_1'])

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

#Shuffle data and build training and test set (75% and 25%)
X_train, X_test, source_type_train, source_type_test, Z_spec_train, Z_spec_test = train_test_split(X, source_type, Z_spec)

#Build an instance of the UMAP algorithm class and use it on the dataset
trans = umap.UMAP().fit(X_train, source_type_train)
train_embedding = trans.embedding_
test_embedding = trans.transform(X_test)

t

#Split the dataset into two different groups
split = 0.93#np.mean(Z_spec)+np.var(Z_spec)**0.5
split_a = Z_spec_test<split
split_b = Z_spec_test>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
my_cmap_1 = copy.copy(plt.cm.summer)
my_cmap_1.set_bad(my_cmap_1(0))
CSa = axs.scatter(test_embedding[:, 0][split_a], test_embedding[:, 1][split_a], 3, c=Z_spec_test[split_a], cmap=my_cmap_1, norm=matplotlib.colors.Normalize(vmin=0,vmax=split))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(test_embedding[:, 0][split_b], test_embedding[:, 1][split_b], 3, c=Z_spec_test[split_b], cmap='autumn_r', norm=matplotlib.colors.Normalize(vmin=split,vmax=np.max(Z_spec)))#, norm=matplotlib.colors.LogNorm(vmin=split,vmax=np.max(zspec_perm)))
cbarb = fig.colorbar(CSb)
cbara.set_label('Z_spec<0.93')
cbarb.set_label('0.93<=Z_spec')
plt.show()

"""
#Object type map

def change_numbers(in_c):
    out_c = np.zeros(len(in_c))
    for i in range(len(in_c)):
        if in_c[i] == -9:
            out_c[i] = 3
        else:
            out_c[i] = in_c[i]
    return out_c

new_source_type_train = change_numbers(source_type_train)
new_source_type_test = change_numbers(source_type_test)

fig, axs = plt.subplots()
custom_cmap = plt.cm.get_cmap('jet', 4)#

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(test_embedding[:, 0], test_embedding[:, 1], s=[new_source_type_test[i]+0.1 for i in range(len(new_source_type_test))], c=new_source_type_test, cmap=custom_cmap)#, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=split))
cbara = fig.colorbar(CSa)
#CSb = axs.scatter(train_embedding[:, 0][split_b], embedding[:, 1][split_b], 1, c=Z_spec[split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(Z_spec)))
#cbarb = fig.colorbar(CSb)
cbara.set_label('Type')
cbara.set_ticks([0.375, 1.5-0.375, 1.5+0.375, 1.5+3*0.375])
cbara.set_ticklabels(['Galaxies', 'Stars', 'X-ray sources', 'No-fit'])
#cbarb.set_label('0.93 <= Z_spec')
plt.show()
"""
