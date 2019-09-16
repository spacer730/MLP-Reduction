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

np.random.seed(33)

#Load the data and reshape it in the form we want
print('Openining the data files')
#fdata_1 = fits.open('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_no_doubles_old_ZCOSMOS_x_COSMOS2015_removed_larger_90.fits')[1].data
#fdata_2 = fits.open('Data/COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_x_COSMOS2015_Laigle+_v1.1_removed_optical_nir_magnitudes_larger_90_no_doubles_C3R2_x_COSMOS2015_90.fits')[1].data
#fdata_3 = fits.open('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_doubles_old_ZCOSMOS_x_COSMOS2015_removed_larger_90.fits')[1].data
fdata_1 = fits.open('Data/C3R2_x_COSMOS2015_no_doubles_old_ZCosmos_plus_CSC2.fits')[1].data
fdata_2 = fits.open('Data/OldZCOSMOS_x_COSMOS2015_no_doubles_C3R2_plus_CSC2.fits')[1].data
fdata_3 = fits.open('Data/C3R2_x_COSMOS2015_doubles_old_ZCosmos_plus_CSC2.fits')[1].data
#fdata_1 = Table.read('Data/C3R2_x_COSMOS2015_no_doubles_old_ZCosmos_plus_CSC2.fits',format='fits')
#fdata_2 = Table.read('Data/OldZCOSMOS_x_COSMOS2015_no_doubles_C3R2_plus_CSC2.fits',format='fits')
#fdata_3 = Table.read('Data/C3R2_x_COSMOS2015_doubles_old_ZCosmos_plus_CSC2.fits',format='fits')
#fdata_1 = Table.read('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_no_doubles_old_ZCOSMOS_x_COSMOS2015_removed_larger_90.fits',format='fits')
#fdata_2 = Table.read('Data/COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_x_COSMOS2015_Laigle+_v1.1_removed_optical_nir_magnitudes_larger_90_no_doubles_C3R2_x_COSMOS2015_90.fits',format='fits')
#fdata_3 = Table.read('Data/KOA_c3r2_v2_COSMOS2015_Laigle+_v1.1_optical_nir_magnitudes_larger_90_doubles_old_ZCOSMOS_x_COSMOS2015_removed_larger_90.fits',format='fits')

print('File is open')

u, B, V, r, ip, zpp = fdata_1['u_MAG_AUTO'], fdata_1['B_MAG_AUTO'], fdata_1['V_MAG_AUTO'], fdata_1['r_MAG_AUTO'], fdata_1['ip_MAG_AUTO'], fdata_1['zpp_MAG_AUTO']
Y, J, H, Ks = fdata_1['Y_MAG_AUTO'], fdata_1['J_MAG_AUTO'], fdata_1['H_MAG_AUTO'], fdata_1['Ks_MAG_AUTO']
ch1, ch2, ch3, ch4 = fdata_1['SPLASH_1_MAG'], fdata_1['SPLASH_2_MAG'], fdata_1['SPLASH_3_MAG'], fdata_1['SPLASH_4_MAG']

zphoto, mass, SSFR, age = fdata_1['PHOTOZ'], fdata_1['MASS_MED'], fdata_1['SSFR_MED'], fdata_1['AGE']
source_type = fdata_1['TYPE']

Z_C3R2 = fdata_1['Redshift']

Xray_data1_indices = np.argwhere(np.array([len(fdata_1['name'][i]) for i in range(len(fdata_1))])>0).flatten()

u, B, V = np.append(u,fdata_2['u_MAG_AUTO']), np.append(B,fdata_2['B_MAG_AUTO']), np.append(V,fdata_2['V_MAG_AUTO'])
r, ip, zpp = np.append(r,fdata_2['r_MAG_AUTO']), np.append(ip,fdata_2['ip_MAG_AUTO']), np.append(zpp,fdata_2['zpp_MAG_AUTO'])
Y, J, H, Ks = np.append(Y,fdata_2['Y_MAG_AUTO']), np.append(J,fdata_2['J_MAG_AUTO']), np.append(H,fdata_2['H_MAG_AUTO']), np.append(Ks,fdata_2['Ks_MAG_AUTO'])
ch1, ch2, ch3, ch4 = np.append(ch1,fdata_2['SPLASH_1_MAG']), np.append(ch2,fdata_2['SPLASH_2_MAG']), np.append(ch3,fdata_2['SPLASH_3_MAG']), np.append(ch4,fdata_2['SPLASH_4_MAG'])

zphoto, mass, SSFR, age = np.append(zphoto,fdata_2['PHOTOZ']), np.append(mass,fdata_2['MASS_MED']), np.append(SSFR,fdata_2['SSFR_MED']), np.append(age,fdata_2['AGE'])
source_type = np.append(source_type,fdata_2['TYPE'])

Z_OldZCOSMOS = fdata_2['zspec']
Instr_OldZCOSMOS = fdata_2['Instr']

Xray_data2_indices = np.argwhere(np.array([len(fdata_2['name'][i]) for i in range(len(fdata_2))])>0).flatten()

u, B, V = np.append(u,fdata_3['u_MAG_AUTO_1']), np.append(B,fdata_3['B_MAG_AUTO_1']), np.append(V,fdata_3['V_MAG_AUTO_1'])
r, ip, zpp = np.append(r,fdata_3['r_MAG_AUTO_1']), np.append(ip,fdata_3['ip_MAG_AUTO_1']), np.append(zpp,fdata_3['zpp_MAG_AUTO_1'])
Y, J, H, Ks = np.append(Y,fdata_3['Y_MAG_AUTO_1']), np.append(J,fdata_3['J_MAG_AUTO_1']), np.append(H,fdata_3['H_MAG_AUTO_1']), np.append(Ks,fdata_3['Ks_MAG_AUTO_1'])
ch1, ch2, ch3, ch4 = np.append(ch1,fdata_3['SPLASH_1_MAG_1']), np.append(ch2,fdata_3['SPLASH_2_MAG_1']), np.append(ch3,fdata_3['SPLASH_3_MAG_1']), np.append(ch4,fdata_3['SPLASH_4_MAG_1'])

zphoto, mass, SSFR, age = np.append(zphoto,fdata_3['PHOTOZ_1']), np.append(mass,fdata_3['MASS_MED_1']), np.append(SSFR,fdata_3['SSFR_MED_1']), np.append(age,fdata_3['AGE_1'])
source_type = np.append(source_type,fdata_3['TYPE_1'])

Z_C3R2_doubles = fdata_3['Redshift']
Z_OldZCOSMOS_doubles = fdata_3['zspec']
Instr_OldZCOSMOS_doubles = fdata_3['Instr']

Xray_data3_indices = np.argwhere(np.array([len(fdata_3['name_1'][i]) for i in range(len(fdata_3))])>0).flatten()

CSC_Xray_indices = np.append(Xray_data1_indices, Xray_data2_indices+len(fdata_1))
CSC_Xray_indices = np.append(CSC_Xray_indices, Xray_data3_indices+len(fdata_1)+len(fdata_2))

COSMOS2015_Xray_indices = np.argwhere(source_type==2).flatten()
union_indices = np.union1d(COSMOS2015_Xray_indices, CSC_Xray_indices)
intersect_indices = np.intersect1d(COSMOS2015_Xray_indices, CSC_Xray_indices)

Z_spec = np.append(Z_C3R2, Z_OldZCOSMOS)
Z_spec = np.append(Z_spec, Z_C3R2_doubles)

#Some interesting lines to analyze the data
np.abs(Z_OldZCOSMOS_doubles-Z_C3R2_doubles)>0.1
source_type[source_type!=0]

u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks = u-B, B-V, V-r, r-ip, ip-zpp, zpp-Y, Y-J, J-H, H-Ks

X = np.array([u, u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks])#, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
X = X.transpose() #This ensures that each array entry are the colors/magnitudes of a galaxy

X[source_type!=1]

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

def indices_neighbors(center_index, eps, source_type_0=True):
    if source_type_0==True:
        extra_sel = source_type==0
    else:
        extra_sel = np.ones(len(source_type), dtype=bool)
    arr = np.argwhere(np.linalg.norm(embedding[extra_sel]-embedding[extra_sel][center_index], axis=1)<eps).flatten()
    return arr[arr!=center_index] #only return the neighbours, not the index itself

def z_umap_local_deviation(index, eps, method=1, source_type_0=True,):
    if source_type_0==True:
        extra_sel = source_type==0
    else:
        extra_sel = np.ones(len(source_type), dtype=bool)
    index_neighbors = indices_neighbors(index, eps, source_type_0)
    z_local_avg = np.mean(Z_spec[extra_sel][index_neighbors])
    if len(index_neighbors)<2:
        return 0.1
    elif method==1:
        deviation = np.abs(Z_spec[extra_sel][index] - z_local_avg)
    elif method==2:
        deviation = np.abs(Z_spec[extra_sel][index] - z_local_avg)/np.var(Z_spec[extra_sel][index_neighbors])
    return deviation

"""
num_neighbors_stype_0 = np.array([len(indices_neighbors(i,0.3, source_type_0=True)) for i in range(len(embedding[source_type==0]))])
deviation_stype_0 = np.array([z_umap_local_deviation(i,method=1,source_type_0=True) for i in range(len(embedding[source_type==0]))])
weighted_deviation_stype_0 = np.array([z_umap_local_deviation(i,method=2,source_type_0=True) for i in range(len(embedding[source_type==0]))])

outlier_criterium = 2
outlier_indices_stype_0 = np.array([i for i in range(len(embedding[source_type==0])) if ((deviation_stype_0[i]>=outlier_criterium)&(num_neighbors_stype_0[i]>10))])

orbiter_indices_stype_0 = [indices_neighbors(outlier_indices_stype_0[i],0.3,source_type_0=True) for i in range(len(outlier_indices_stype_0))]
orbiter_indices_stype_0_flat = np.concatenate(orbiter_indices_stype_0).ravel()

outlier2_criterium = 200
outlier2_indices_stype_0 = np.array([i for i in range(len(embedding[source_type==0])) if ((weighted_deviation_stype_0[i]>=outlier2_criterium)&(num_neighbors_stype_0[i]>10))])

orbiter2_indices_stype_0 = [indices_neighbors(outlier2_indices_stype_0[i],0.3,source_type_0=True) for i in range(len(outlier2_indices_stype_0))]
orbiter2_indices_stype_0_flat = np.concatenate(orbiter2_indices_stype_0).ravel()

crossmatched_indices = [index for index in outlier_indices_stype_0 if index in CSC_Xray_indices]
crossmatched_indices_2 = [index for index in outlier2_indices_stype_0 if index in CSC_Xray_indices]

#Now repeat all of that for the dataset with all sourcetypes included
num_neighbors_stype_all = np.array([len(indices_neighbors(i,0.3, source_type_0=False)) for i in range(len(embedding))])
deviation_stype_all = np.array([z_umap_local_deviation(i,0.3,method=1,source_type_0=False) for i in range(len(embedding))])
weighted_deviation_stype_all = np.array([z_umap_local_deviation(i,0.3,method=2,source_type_0=False) for i in range(len(embedding))])

outlier_criterium = 1.5
outlier_indices_stype_all = np.array([i for i in range(len(embedding)) if ((deviation_stype_all[i]>=outlier_criterium)&(num_neighbors_stype_all[i]>10))])

orbiter_indices_stype_all = [indices_neighbors(outlier_indices_stype_all[i],0.3,source_type_0=False) for i in range(len(outlier_indices_stype_all))]
orbiter_indices_stype_all_flat = np.concatenate(orbiter_indices_stype_all).ravel()

outlier2_criterium = 100
outlier2_indices_stype_all = np.array([i for i in range(len(embedding)) if ((weighted_deviation_stype_all[i]>=outlier2_criterium)&(num_neighbors_stype_all[i]>10))])

orbiter2_indices_stype_all = [indices_neighbors(outlier2_indices_stype_all[i],0.3,source_type_0=False) for i in range(len(outlier2_indices_stype_all))]
orbiter2_indices_stype_all_flat = np.concatenate(orbiter2_indices_stype_all).ravel()

crossmatched_indices_3 = [index for index in outlier_indices_stype_all if index in COSMOS2015_Xray_indices]
print("The number of outliers is "+str(len(outlier_indices_stype_all))+" and "+str(len(crossmatched_indices_3))+" of them are in the CSC catalog (selected using redshift deviation).")
crossmatched_indices_4 = [index for index in outlier2_indices_stype_all if index in COSMOS2015_Xray_indices]
print("The number of outliers is "+str(len(outlier2_indices_stype_all))+" and "+str(len(crossmatched_indices_4))+" of them are in the CSC catalog (selected using weighted redshift deviation).")
"""

def outliers(eps, outlier_criteria, outlier2_criteria, top_percent=False, data_return=False):
    num_neighbors_stype_all = np.array([len(indices_neighbors(i,eps, source_type_0=False)) for i in range(len(embedding))])
    deviation_stype_all = np.array([z_umap_local_deviation(i,eps,method=1,source_type_0=False) for i in range(len(embedding))])
    weighted_deviation_stype_all = np.array([z_umap_local_deviation(i,eps,method=2,source_type_0=False) for i in range(len(embedding))])
    #
    results_outliers = []
    results_outliers2 = []
    if top_percent == False:
        for k in range(len(outlier_criteria)):
            outlier_indices_stype_all = np.array([i for i in range(len(embedding)) if ((deviation_stype_all[i]>=outlier_criteria[k])&(num_neighbors_stype_all[i]>5))])
            crossmatched_indices = [index for index in outlier_indices_stype_all if index in CSC_Xray_indices]
            results_outliers.append((len(outlier_indices_stype_all),len(crossmatched_indices)))
        #
        for j in range(len(outlier_criteria)):
            outlier2_indices_stype_all = np.array([i for i in range(len(embedding)) if ((weighted_deviation_stype_all[i]>=outlier2_criteria[j])&(num_neighbors_stype_all[i]>5))])
            crossmatched_indices_2 = [index for index in outlier2_indices_stype_all if index in CSC_Xray_indices]
            results_outliers2.append((len(outlier2_indices_stype_all),len(crossmatched_indices_2)))
    elif top_percent == True:
        top_211_outlier_indices_stype_all = sorted(range(len(deviation_stype_all)), key=lambda i: deviation_stype_all[i])[-211:]
        top_211_crossmatched_indices = [index for index in top_211_outlier_indices_stype_all if index in CSC_Xray_indices]
        top_21_outlier_indices_stype_all = sorted(range(len(deviation_stype_all)), key=lambda i: deviation_stype_all[i])[-21:]
        top_21_crossmatched_indices = [index for index in top_21_outlier_indices_stype_all if index in CSC_Xray_indices]
        #
        top_211_outlier2_indices_stype_all = sorted(range(len(weighted_deviation_stype_all)), key=lambda i: weighted_deviation_stype_all[i])[-211:]
        top_211_crossmatched_indices_2 = [index for index in top_211_outlier2_indices_stype_all if index in CSC_Xray_indices]
        top_21_outlier2_indices_stype_all = sorted(range(len(weighted_deviation_stype_all)), key=lambda i: weighted_deviation_stype_all[i])[-21:]
        top_21_crossmatched_indices_2 = [index for index in top_21_outlier2_indices_stype_all if index in CSC_Xray_indices]
        #
        if data_return == False:
            return (len(top_211_outlier_indices_stype_all),len(top_211_crossmatched_indices)),(len(top_21_outlier_indices_stype_all),len(top_21_crossmatched_indices)) \
                   ,(len(top_211_outlier2_indices_stype_all),len(top_211_crossmatched_indices_2)),(len(top_21_outlier2_indices_stype_all),len(top_21_crossmatched_indices_2))
    #
    if data_return == True:
        return top_211_outlier_indices_stype_all, top_21_outlier_indices_stype_all, top_211_outlier2_indices_stype_all, top_21_outlier2_indices_stype_all
    elif data_return == False:
        return results_outliers, results_outliers2
    #[(len(outlier_indices_stype_all), len(crossmatched_indices)), (len(outlier2_indices_stype_all), len(crossmatched_indices_2))]

"""
Some parameter space search:

outlier_criteria = np.arange(1,2.5,0.25)
outlier2_criteria = np.arange(75,225,25)
epsila = np.arange(0.1,1.2,0.2)
outlier_results = [[] for i in range(len(epsila))]

for j in range(len(epsila)):
    outlier_results[j].append(outliers(epsila[j],outlier_criteria,outlier2_criteria))
    #for k in range(len(outlier_criteria):
        #outlier_results[j].append(outliers(epsila[j],outlier_criteria[k],outlier2_criteria[k]))

epsila2 = np.arange(1.2,2.2,0.2)
outlier_results2 = [[] for i in range(len(epsila))]

for j in range(len(epsila2)):
    outlier_results2[j].append(outliers(epsila2[j],outlier_criteria,outlier2_criteria))

outlier_criteria2 = np.linspace(1,1.5,6)
outlier2_criteria2 = np.linspace(25,75,6)
epsila3 = np.linspace(0.1,0.5,5)
outlier_results3 = [[] for i in range(len(epsila3))]

for j in range(len(epsila3)):
    outlier_results3[j].append(outliers(epsila3[j],outlier_criteria2,outlier2_criteria2))

outlier_criteria3 = np.linspace(2.5,3.75,6)
outlier2_criteria3 = np.linspace(25,75,6)
epsila4 = epsila2
outlier_results4 = [[] for i in range(len(epsila4))]

for j in range(len(epsila4)):
    outlier_results4[j].append(outliers(epsila4[j],outlier_criteria3,outlier2_criteria3))

epsila5 = np.linspace(0.2,2,10)
outlier_results5 = []
for j in range(len(epsila5)):
    outlier_results5.append(outliers(epsila5[j],1,1,top_percent=True))

epsila6 = np.linspace(2.2,3,5)
outlier_results6 = []
for j in range(len(epsila6)):
    outlier_results6.append(outliers(epsila6[j],1,1,top_percent=True))

epsila7 = np.linspace(3.2,5,5)
outlier_results7 = []
for j in range(len(epsila7)):
    outlier_results7.append(outliers(epsila7[j],1,1,top_percent=True))
"""

indices_top_arr = outliers(0.6, 1, 1, top_percent=True, data_return=True)
indices_top_21_redshift_deviation = indices_top_arr[1]
indices_top_211_redshift_deviation = indices_top_arr[0]
indices_top_21_weighted_redshift_deviation = outliers(1.8, 1, 1, top_percent=True, data_return=True)[3]
indices_top_211_weighted_redshift_deviation = outliers(3.2, 1, 1, top_percent=True, data_return=True)[2]
    
def get_data_indices(indices_outliers):
    indices_outliers_1 = np.array([])
    indices_outliers_2 = np.array([])
    indices_outliers_3 = np.array([])
    #
    for index in indices_outliers:
        if (0 <= index) & (index <= 2101):
            indices_outliers_1 = np.append(indices_outliers_1, index)
        
        elif (index <= 20886):
            indices_outliers_2 = np.append(indices_outliers_2, index-2102)
        
        elif (index <= 21113):
            indices_outliers_3 = np.append(indices_outliers_3, index-20887)
    #
    data_outliers_1 = fdata_1[indices_outliers_1.astype(int)]
    data_outliers_2 = fdata_2[indices_outliers_2.astype(int)]
    data_outliers_3 = fdata_3[indices_outliers_3.astype(int)]
    new_hdu_1 = fits.BinTableHDU.from_columns(data_outliers_1._get_raw_data())
    new_hdu_2 = fits.BinTableHDU.from_columns(data_outliers_2._get_raw_data())
    new_hdu_3 = fits.BinTableHDU.from_columns(data_outliers_3._get_raw_data())
    new_hdu_1.writeto("Data/Outliers/C3R2_x_COSMOS2015_no_doubles_old_ZCosmos_plus_CSC2_211_outliers_eps_3dot2.fits")
    new_hdu_2.writeto("Data/Outliers/OldZCOSMOS_x_COSMOS2015_no_doubles_C3R2_plus_CSC2_211_outliers_eps_3dot2.fits")
    new_hdu_3.writeto("Data/Outliers/C3R2_x_COSMOS2015_doubles_old_ZCosmos_plus_CSC2_211_outliers_eps_3dot2.fits")

"""
#Normal Map
#Split the dataset into two different groups
split = 1.47#np.mean(Z_spec)+np.var(Z_spec)**0.5
split_a = np.argwhere(Z_spec[source_type==0]<split).flatten()
split_b = np.argwhere(Z_spec[source_type==0]>=split).flatten()

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
my_cmap_1 = copy.copy(plt.cm.summer)
my_cmap_1.set_bad(my_cmap_1(0))
CSa = axs.scatter(embedding[:, 0][source_type==0][split_a], embedding[:, 1][source_type==0][split_a], s=1, c=Z_spec[source_type==0][split_a], cmap=my_cmap_1, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=split))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][source_type==0][split_b], embedding[:, 1][source_type==0][split_b], s=1, c=Z_spec[source_type==0][split_b], cmap='autumn_r', norm=matplotlib.colors.Normalize(vmin=split, vmax=np.max(Z_spec)))
cbarb = fig.colorbar(CSb)
#axs.text(2.5,8, 'Source_type == 0, outliers')
cbara.set_label('Z_spec < 1.47')
cbarb.set_label('1.47 <= Z_spec')
#axs.set_xlim([-9.75,8.8])
#axs.set_ylim([-10.4,8.7])
plt.show()
"""

#Shows whole projection with the outliers enlarged
#Split the dataset into two different groups
split = 1.47#np.mean(Z_spec)+np.var(Z_spec)**0.5
split_a = np.argwhere(Z_spec<split).flatten()
split_b = np.argwhere(Z_spec>=split).flatten()

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
my_cmap_1 = copy.copy(plt.cm.summer)
my_cmap_1.set_bad(my_cmap_1(0))
CSa = axs.scatter(embedding[:, 0][split_a], embedding[:, 1][split_a], s=[30 if np.any(i==CSC_Xray_indices) else 1 for i in split_a], c=Z_spec[split_a], cmap=my_cmap_1, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=split))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], s=[30 if np.any(i==CSC_Xray_indices) else 1 for i in split_b], c=Z_spec[split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(Z_spec)))
cbarb = fig.colorbar(CSb)
#axs.text(2.5,8, 'Source_type == 0, outliers')
cbara.set_label('Z_spec < 1.47')
cbarb.set_label('1.47 <= Z_spec')
#axs.set_xlim([-9.75,8.8])
#axs.set_ylim([-10.4,8.7])
plt.show()

"""
#Shows the outliers numbered as well as an island of their closest neighbors surrounding them
#Split the dataset into two different groups
split = 1.47#np.mean(Z_spec)+np.var(Z_spec)**0.5
split_a = Z_spec[source_type==0][np.append(outlier2_indices,orbiter2_indices_flat)]<split
split_b = Z_spec[source_type==0][np.append(outlier2_indices,orbiter2_indices_flat)]>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

for i in range(len(outlier2_indices)):
    axs.text(embedding[:,0][source_type == 0][outlier2_indices[i]], embedding[:,1][source_type == 0][outlier2_indices[i]], str(i+1), color = 'k')

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(embedding[:, 0][source_type==0][np.append(outlier2_indices,orbiter2_indices_flat)][split_a], embedding[:, 1][source_type==0][np.append(outlier2_indices,orbiter2_indices_flat)][split_a], s=1, c=Z_spec[source_type==0][np.append(outlier2_indices,orbiter2_indices_flat)][split_a], cmap='summer', norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=split))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][source_type==0][np.append(outlier2_indices,orbiter2_indices_flat)][split_b], embedding[:, 1][source_type==0][np.append(outlier2_indices,orbiter2_indices_flat)][split_b], s=1, c=Z_spec[source_type==0][np.append(outlier2_indices,orbiter2_indices_flat)][split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(Z_spec)))
cbarb = fig.colorbar(CSb)
axs.text(2.5,8, 'Source_type == 0, outliers & neighbors')
cbara.set_label('Z_spec < 0.93')
cbarb.set_label('0.93 <= Z_spec')
#axs.set_xlim([-9.75,8.8])
#axs.set_ylim([-10.4,8.7])
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

#new_source_type = change_numbers(source_type)
new_source_type = np.array([1 if np.any(i==CSC_Xray_indices) else 0 for i in range(len(Z_spec))])

fig, axs = plt.subplots()
custom_cmap = plt.cm.get_cmap('bwr', 2)#

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(embedding[:, 0], embedding[:, 1], s=np.array([0.5+5*new_source_type[i] for i in range(len(new_source_type))]), c=new_source_type, cmap=custom_cmap)#, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=split))
cbara = fig.colorbar(CSa)
#CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], 1, c=Z_spec[split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(Z_spec)))
#cbarb = fig.colorbar(CSb)
cbara.set_label('Type')
cbara.set_ticks([0.25, 0.75])#, 1.5+0.375, 1.5+3*0.375])
cbara.set_ticklabels(['Not in CSC', 'In CSC'])#, 'Stars', 'X-ray sources', 'No-fit'])
#cbarb.set_label('0.93 <= Z_spec')
plt.show()
