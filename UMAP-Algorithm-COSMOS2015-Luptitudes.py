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
fdata = fits.open('Data/COSMOS2015_Laigle+_v1.1_removed_optical_nir_magnitudes_larger_90.fits')[1].data

zphoto = fdata['PHOTOZ']
mass = fdata['MASS_MED']
SSFR = fdata['SSFR_MED']
age = fdata['AGE']

def zeropoint(band):
    band_mag_aper2 = fdata[band+'_MAG_APER2']
    band_flux_aper2 = fdata[band+'_FLUX_APER2']

    band_mag_aper3 = fdata[band+'_MAG_APER3']
    band_flux_aper3 = fdata[band+'_FLUX_APER3']

    zeropoint_band_magnitude_aper2_0 = band_mag_aper2[0] + 2.5*np.log10(band_flux_aper2[0])
    zeropoint_band_magnitude_aper2_1 = band_mag_aper2[1] + 2.5*np.log10(band_flux_aper2[1])

    zeropoint_band_magnitude_aper3_0 = band_mag_aper3[0] + 2.5*np.log10(band_flux_aper3[0])
    zeropoint_band_magnitude_aper3_1 = band_mag_aper3[1] + 2.5*np.log10(band_flux_aper3[1])

    return (zeropoint_band_magnitude_aper2_0, zeropoint_band_magnitude_aper2_1, zeropoint_band_magnitude_aper3_0, zeropoint_band_magnitude_aper3_1)

u_zeropoints = zeropoint('u')
print('u_zeropoints are: ' + str(u_zeropoints))
B_zeropoints = zeropoint('B')
print('B_zeropoints are: ' + str(B_zeropoints))
V_zeropoints = zeropoint('V')
print('V_zeropoints are: ' + str(V_zeropoints))
r_zeropoints = zeropoint('r')
print('r_zeropoints are: ' + str(r_zeropoints))
ip_zeropoints = zeropoint('ip')
print('ip_zeropoints are: ' + str(ip_zeropoints))
zpp_zeropoints = zeropoint('zpp')
print('zpp_zeropoints are: ' + str(zpp_zeropoints))
yHSC_zeropoints = zeropoint('yHSC')
print('yHSC_zeropoints are: ' + str(yHSC_zeropoints))
Y_zeropoints = zeropoint('Y')
print('Y_zeropoints are: ' + str(Y_zeropoints))
J_zeropoints = zeropoint('J')
print('J_zeropoints are: ' + str(J_zeropoints))
H_zeropoints = zeropoint('H')
print('H_zeropoints are: ' + str(H_zeropoints))
Ks_zeropoints = zeropoint('Ks')
print('Ks_zeropoints are: ' + str(Ks_zeropoints))

def AUTO_FLUX(band):
    return 10**(-(fdata[band+'_MAG_AUTO']-23.9)/2.5)

u_FLUX_AUTO = AUTO_FLUX('u')
B_FLUX_AUTO = AUTO_FLUX('B')
V_FLUX_AUTO = AUTO_FLUX('V')
r_FLUX_AUTO = AUTO_FLUX('r')
ip_FLUX_AUTO = AUTO_FLUX('ip')
zpp_FLUX_AUTO = AUTO_FLUX('zpp')
yHSC_FLUX_AUTO = AUTO_FLUX('yHSC')
Y_FLUX_AUTO = AUTO_FLUX('Y')
J_FLUX_AUTO = AUTO_FLUX('J')
H_FLUX_AUTO = AUTO_FLUX('H')
Ks_FLUX_AUTO = AUTO_FLUX('Ks')

def luptitude(fluxes, zeropoint_mag):
    a = 2.5
    b = 1.042*(np.var(fluxes)**0.5)
    zeropoint_luptitude = zeropoint_mag - 2.5*np.log10(b)
    luptitudes = zeropoint_luptitude - a*np.arcsinh(fluxes/(2*b))
    return luptitudes

u_LUPTITUDE_AUTO = luptitude(u_FLUX_AUTO, 23.9)
B_LUPTITUDE_AUTO = luptitude(B_FLUX_AUTO, 23.9)
V_LUPTITUDE_AUTO = luptitude(V_FLUX_AUTO, 23.9)
r_LUPTITUDE_AUTO = luptitude(r_FLUX_AUTO, 23.9)
ip_LUPTITUDE_AUTO = luptitude(ip_FLUX_AUTO, 23.9)
zpp_LUPTITUDE_AUTO = luptitude(zpp_FLUX_AUTO, 23.9)
yHSC_LUPTITUDE_AUTO = luptitude(yHSC_FLUX_AUTO, 23.9)
Y_LUPTITUDE_AUTO = luptitude(Y_FLUX_AUTO, 23.9)
J_LUPTITUDE_AUTO = luptitude(J_FLUX_AUTO, 23.9)
H_LUPTITUDE_AUTO = luptitude(H_FLUX_AUTO, 23.9)
Ks_LUPTITUDE_AUTO = luptitude(Ks_FLUX_AUTO, 23.9)

u_LUPTITUDE_APER2 = luptitude(fdata['u_FLUX_APER2'], 23.9)
B_LUPTITUDE_APER2 = luptitude(fdata['B_FLUX_APER2'], 23.9)
V_LUPTITUDE_APER2 = luptitude(fdata['V_FLUX_APER2'], 23.9)
r_LUPTITUDE_APER2 = luptitude(fdata['r_FLUX_APER2'], 23.9)
ip_LUPTITUDE_APER2 = luptitude(fdata['ip_FLUX_APER2'], 23.9)
zpp_LUPTITUDE_APER2 = luptitude(fdata['zpp_FLUX_APER2'], 23.9)
yHSC_LUPTITUDE_APER2 = luptitude(fdata['yHSC_FLUX_APER2'], 23.9)
Y_LUPTITUDE_APER2 = luptitude(fdata['Y_FLUX_APER2'], 23.9)
J_LUPTITUDE_APER2 = luptitude(fdata['J_FLUX_APER2'], 23.9)
H_LUPTITUDE_APER2 = luptitude(fdata['H_FLUX_APER2'], 23.9)
Ks_LUPTITUDE_APER2 = luptitude(fdata['Ks_FLUX_APER2'], 23.9)

u_LUPTITUDE_APER3 = luptitude(fdata['u_FLUX_APER3'], 23.9)
B_LUPTITUDE_APER3 = luptitude(fdata['B_FLUX_APER3'], 23.9)
V_LUPTITUDE_APER3 = luptitude(fdata['V_FLUX_APER3'], 23.9)
r_LUPTITUDE_APER3 = luptitude(fdata['r_FLUX_APER3'], 23.9)
ip_LUPTITUDE_APER3 = luptitude(fdata['ip_FLUX_APER3'], 23.9)
zpp_LUPTITUDE_APER3 = luptitude(fdata['zpp_FLUX_APER3'], 23.9)
yHSC_LUPTITUDE_APER3 = luptitude(fdata['yHSC_FLUX_APER3'], 23.9)
Y_LUPTITUDE_APER3 = luptitude(fdata['Y_FLUX_APER3'], 23.9)
J_LUPTITUDE_APER3 = luptitude(fdata['J_FLUX_APER3'], 23.9)
H_LUPTITUDE_APER3 = luptitude(fdata['H_FLUX_APER3'], 23.9)
Ks_LUPTITUDE_APER3 = luptitude(fdata['Ks_FLUX_APER3'], 23.9)

u_ip_LUPTICOLOR_AUTO = u_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
B_ip_LUPTICOLOR_AUTO = B_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
V_ip_LUPTICOLOR_AUTO = V_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
r_ip_LUPTICOLOR_AUTO = B_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
zpp_ip_LUPTICOLOR_AUTO = zpp_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
yHSC_ip_LUPTICOLOR_AUTO = yHSC_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
Y_ip_LUPTICOLOR_AUTO = Y_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
J_ip_LUPTICOLOR_AUTO = J_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
H_ip_LUPTICOLOR_AUTO = H_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO
Ks_ip_LUPTICOLOR_AUTO = Ks_LUPTITUDE_AUTO - ip_LUPTITUDE_AUTO

X = np.array([u_ip_LUPTICOLOR_AUTO, B_ip_LUPTICOLOR_AUTO, V_ip_LUPTICOLOR_AUTO, r_ip_LUPTICOLOR_AUTO, zpp_ip_LUPTICOLOR_AUTO, Y_ip_LUPTICOLOR_AUTO, J_ip_LUPTICOLOR_AUTO, H_ip_LUPTICOLOR_AUTO, Ks_ip_LUPTICOLOR_AUTO])
X = X.transpose() #This makes it so that each array entry is all band magnitudes of a galaxy

#Shuffle data and build training and test set
permuted_indices = np.random.permutation(len(X))
X_perm = X[permuted_indices]
#X_perm = X_perm.reshape((np.shape(X_perm)[0],np.shape(X_perm)[-1]))
zphoto_perm = zphoto[permuted_indices]#.reshape(-1)
mass_perm = mass[permuted_indices]#.reshape(-1)
SSFR_perm = SSFR[permuted_indices]#.reshape(-1)
age_perm = age[permuted_indices]#.reshape(-1)

#Remove problematic galaxies
good_indices = np.argwhere((np.abs(zphoto_perm)<90) & (np.abs(mass_perm)<90) & (np.abs(SSFR_perm)<90) & (age_perm>100)).flatten() #& (np.abs(X_perm[:,-1])<=90) & (np.abs(X_perm[:,-2])<=90)).flatten()

X_perm = X_perm[good_indices]
zphoto_perm = zphoto_perm[good_indices]
mass_perm = mass_perm[good_indices]
SSFR_perm = SSFR_perm[good_indices]
age_perm = age_perm[good_indices]

#Build an instance of the UMAP algorithm class and use it on the dataset
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_perm)

#Split the dataset into two different groups
split = np.mean(SSFR_perm)-np.var(SSFR_perm)**0.5
split_a = SSFR_perm<split
split_b = SSFR_perm>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
#CSa = axs.scatter(embedding[:, 0][split_a], embedding[:, 1][split_a], 3, c=SSFR_perm[split_a], cmap='summer')#, norm=matplotlib.colors.LogNorm())
#cbara = fig.colorbar(CSa)
#axs.text(5,6,'0.9<=z<1.1',color='k')
CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 3, c=SSFR_perm[split_b], cmap='autumn_r')#, norm=matplotlib.colors.LogNorm())
cbarb = fig.colorbar(CSb)
#cbara.set_label('SSFR < -9.83')
cbarb.set_label('-9.83 <= SSFR')
plt.show()
