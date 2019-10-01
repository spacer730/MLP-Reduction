import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
import matplotlib.colors
import matplotlib as mpl
import copy
import sklearn
from astropy.io import fits
from astropy.table import Table
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

np.random.seed(33)

print('Openining the data files')
fdata_1 = fits.open('Data/C3R2_x_COSMOS2015_no_doubles_old_ZCosmos_plus_CSC2.fits')[1].data
fdata_2 = fits.open('Data/OldZCOSMOS_x_COSMOS2015_no_doubles_C3R2_plus_CSC2.fits')[1].data
fdata_3 = fits.open('Data/C3R2_x_COSMOS2015_doubles_old_ZCosmos_plus_CSC2.fits')[1].data

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
X = X[0:10000]

obj_ids = np.arange(len(X))

"""
## Outlier detection algorithm
We wish to detect the outlying objects in the given 2D data from the previous cell (which we call *'real'*). We will do that step by step in the following cells.

**Step 1:** create synthetic data with the same sample size as the *'real'* data, and the same marginal distributions in its features. We call these data *'synthetic'*.

We build the *'synthetic'* data with **the same size** as the *'real'* data because we want the RF to train on a balanced sample.
That is, the RF performs better when the different classes have approximatly the same number of objects.
Otherwise, the trained forest will perform better on the bigger class and worse on the smaller class.

We build the *'synthetic'* data with the same marginal distributions, because we wish to detect objects with outlying covariance (and higher moments).
We show in the paper that this method works best for anomaly detection on galaxy spectra. Other possible choices are discussed by Shi & Horvath (2006).
"""

def return_synthetic_data(X):
    """
    The function returns a matrix with the same dimensions as X but with synthetic data
    based on the marginal distributions of its featues
    """
    features = len(X[0])
    X_syn = np.zeros(X.shape)
    #
    for i in range(features):
        obs_vec = X[:,i]
        syn_vec = np.random.choice(obs_vec, len(obs_vec)) # here we chose the synthetic data to match the marginal distribution of the real data
        X_syn[:,i] += syn_vec
    #
    return X_syn

X_syn = return_synthetic_data(X)

#Show differences between the correlation of certain features for the real data and the synthetic data
plt.rcParams['figure.figsize'] = 8, 4

plt.subplot(1, 2, 1)
plt.title("real data")
plt.plot(X[:, 3], X[:, 4], "ok", markersize=3)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 2, 2)
plt.title("synthetic data")
plt.plot(X_syn[:, 3], X_syn[:, 4], "og", markersize=3)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

def merge_work_and_synthetic_samples(X, X_syn):
    """
    The function merges the data into one sample, giving the label "1" to the real data and label "2" to the synthetic data
    """
    # build the labels vector
    Y = np.ones(len(X))
    Y_syn = np.ones(len(X_syn)) * 2
    #
    Y_total = np.concatenate((Y, Y_syn))
    X_total = np.concatenate((X, X_syn))
    return X_total, Y_total

X_total, Y_total = merge_work_and_synthetic_samples(X, X_syn)
# declare an RF
N_TRAIN = 500 # number of trees in the forest
rand_f = RandomForestClassifier(n_estimators=N_TRAIN)
rand_f.fit(X_total, Y_total)

def build_similarity_matrix(rand_f, X):
    """
    The function builds the similarity matrix based on the feature matrix X for the results Y
    based on the random forest we've trained
    the matrix is normalised so that the biggest similarity is 1 and the lowest is 0

    This function counts only leaves in which the object is classified as a "real" object 
    it is also implemented to optimize running time, asumming one has enough running memory
    """
    # apply to get the leaf indices
    apply_mat = rand_f.apply(X)
    # find the predictions of the sample
    is_good_matrix = np.zeros(apply_mat.shape)
    for i, est in enumerate(rand_f.estimators_):
        d = est.predict_proba(X)[:, 0] == 1
        is_good_matrix[:, i] = d
    # mark leaves that make the wrong prediction as -1, in order to remove them from the distance measurement
    apply_mat[is_good_matrix == False] = -1
    # now calculate the similarity matrix
    sim_mat = np.sum((apply_mat[:, None] == apply_mat[None, :]) & (apply_mat[:, None] != -1) & (apply_mat[None, :] != -1), axis=2) / np.asfarray(np.sum([apply_mat != -1], axis=2), dtype='float')
    return sim_mat

sim_mat = build_similarity_matrix(rand_f, X)
dis_mat = 1 - sim_mat

sum_vec = np.sum(dis_mat, axis=1)
sum_vec /= float(len(sum_vec))

plt.rcParams['figure.figsize'] = 6, 4
plt.title("Weirdness score histogram")
tmp = plt.hist(sum_vec, bins=60, color="g")
plt.ylabel("N")
plt.xlabel("weirdness score")
plt.show()

sum_vec_21_outliers = np.sort(sum_vec)[::-1][:21]
obj_ids_21_outliers = obj_ids[np.argsort(sum_vec)][::-1][:21]

sum_vec_211_outliers = np.sort(sum_vec)[::-1][:211]
obj_ids_211_outliers = obj_ids[np.argsort(sum_vec)][::-1][:211]

sum_vec_5000_outliers = np.sort(sum_vec)[::-1][:5000]
obj_ids_5000_outliers = obj_ids[np.argsort(sum_vec)][::-1][:5000]

plt.rcParams['figure.figsize'] = 5, 5
plt.title("Data and outliers")
plt.plot(X[:,0], X[:,1], "ok", label="input daya", markersize=4)
plt.plot(X[obj_ids_outliers, 0], X[obj_ids_outliers, 1], "om", label="outliers", markersize=4)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")	
plt.legend(loc="best")
plt.show()
