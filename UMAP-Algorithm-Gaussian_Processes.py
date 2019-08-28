import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
import matplotlib as mpl
from matplotlib import cm
import os
from astropy.io import fits
from astropy.table import Table
from sklearn.model_selection import train_test_split
import umap
import george
from scipy.optimize import minimize
from george import kernels

np.random.seed(33)

#Load the data en reshape it in the form we want
print('Openining the data file')
fdata = fits.open('Data/COSMOS2015_Laigle+_v1.1_removed_optical_nir_magnitudes_larger_90.fits')[1].data
print('File is open')

u, B, V, r, ip, zpp = fdata['u_MAG_AUTO'], fdata['B_MAG_AUTO'], fdata['V_MAG_AUTO'], fdata['r_MAG_AUTO'], fdata['ip_MAG_AUTO'], fdata['zpp_MAG_AUTO']
Y, J, H, Ks = fdata['Y_MAG_AUTO'], fdata['J_MAG_AUTO'], fdata['H_MAG_AUTO'], fdata['Ks_MAG_AUTO']
ch1, ch2, ch3, ch4 = fdata['SPLASH_1_MAG'], fdata['SPLASH_2_MAG'], fdata['SPLASH_3_MAG'], fdata['SPLASH_4_MAG']

zphoto, mass, SSFR, age = fdata['PHOTOZ'], fdata['MASS_MED'], fdata['SSFR_MED'], fdata['AGE']
source_type = fdata['type']

#mass_med_min68, mass_med_max68 = fdata['MASS_MED_MIN68'], fdata['MASS_MED_MAX68']

u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks = u-B, B-V, V-r, r-ip, ip-zpp, zpp-Y, Y-J, J-H, H-Ks

X = np.array([u, u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks])#, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
X = X.transpose() #This makes it so that each array entry is all band magnitudes of a galaxy

#Select only object type galaxy and remove problematic galaxies
good_indices = np.argwhere((np.abs(zphoto)<90) & (np.abs(mass)<90) & (np.abs(SSFR)<90) & (age>100) & (source_type == 0)).flatten() #& (np.abs(X[:,-1])<=90) & (np.abs(X[:,-2])<=90)).flatten()
X, zphoto, mass, SSFR, age= X[good_indices], zphoto[good_indices], mass[good_indices], SSFR[good_indices], age[good_indices]

#Shuffle data and build training and test set (75% and 25%)
X_train, X_test, zphoto_train, zphoto_test, mass_train, mass_test, SSFR_train, SSFR_test, age_train, age_test = train_test_split(X, zphoto, mass, SSFR, age)

#Build an instance of the UMAP algorithm class and use it on the dataset
trans = umap.UMAP().fit(X_train)
train_embedding = trans.embedding_
test_embedding = trans.transform(X_test)

#Take data from bin out of UMAP
binned_indices_train = np.argwhere((zphoto_train>=0.7) & (zphoto_train<0.9)).flatten()
X_train_bin = X_train[binned_indices_train]
train_embedding_bin = train_embedding[binned_indices_train]
zphoto_train_bin, mass_train_bin, SSFR_train_bin, age_train_bin = zphoto_train[binned_indices_train], mass_train[binned_indices_train], SSFR_train[binned_indices_train], age_train[binned_indices_train]

binned_indices_test = np.argwhere((zphoto_test>=0.7) & (zphoto_test<0.9)).flatten()
X_test_bin = X_test[binned_indices_test]
test_embedding_bin = test_embedding[binned_indices_test]
zphoto_test_bin, mass_test_bin, SSFR_test_bin, age_test_bin = zphoto_test[binned_indices_test], mass_test[binned_indices_test], SSFR_test[binned_indices_test], age_test[binned_indices_test]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#Training GP regressor
y = mass_train_bin
x = train_embedding_bin
x_pred = test_embedding_bin

kernel = np.var(y) * kernels.ExpSquaredKernel(0.5, ndim=2)
gp = george.GP(kernel)
gp.compute(x)

pred, pred_var = gp.predict(y, x_pred, return_var=True)

def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)

result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
print(result)

gp.set_parameter_vector(result.x)
print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))

pred, pred_var = gp.predict(y, x_pred, return_var=True)

"""
#Hexbin scatter plot of M_pred vs M_true (SED-fit)
fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(2, 1, height_ratios=[4,1], hspace=0.03) 

ax0 = plt.subplot(gs[0])
hb = ax0.hexbin(mass_test_bin,mass_test_pred_2[9],vmin=1, gridsize=100, bins='log', cmap='plasma')
cb = fig.colorbar(hb, ax=ax0)
cb.set_label('N')
ax0.set_ylabel(r'$\rm Log\ (M_{*}/M_{\odot}) \ knn-10 \ Input \ Space$',fontsize=20)
ax0.plot([4,12],[4,12],'k--',linewidth=1)
ax0.set_xlim([7,np.max(mass_test_bin)])
ax0.set_ylim([7,np.max(mass_test_bin)])
ax0.text(10.2,7.7,r'$\rm 0.7<z<0.9$',size=15,color='k')
ax0.axes.get_xaxis().set_ticks([])
plt.tick_params(axis='both', which='major', labelsize=13)

ax2 = plt.subplot(gs[1])
ax2.set_xlabel(r'$\rm Log\ (M_{*}/M_{\odot})\ COSMOS\ SEDfit$',fontsize=20, labelpad=10)
ax2.set_ylabel(r'$\rm \Delta \ Log(M)$',fontsize=20,labelpad=10)
ax2.set_ylim([-0.5,0.5])
hb=ax2.hexbin(mass_test_bin,mass_test_bin-mass_test_pred_2[9],vmin=1, gridsize=100, bins='log', cmap='plasma')
cb = fig.colorbar(hb, ax=ax2)
cb.set_label('N')
ax2.plot([6,12],[0,0],'k--')
ax2.set_xlim([7,np.max(mass_test_bin)])
ax2.set_ylim([-1.,1.])
ax2.yaxis.set_ticks([-1.0,-0.5,0.,0.5,1.0])
plt.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
plt.show()
"""

"""
#Split the dataset into two different groups
split = 3*10**9#np.mean(age_train)+np.var(age_train)**0.5
split_a = age_train<split
split_b = age_train>=split

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()

#x,y coordinates and the size of the dot and whether to use a logscale for the colors
CSa = axs.scatter(train_embedding[:, 0][split_a], train_embedding[:, 1][split_a], 0.01, c=age_train[split_a], cmap='summer', norm=matplotlib.colors.LogNorm(vmin=np.min(age_train),vmax=split))
cbara = fig.colorbar(CSa)
#axs.text(5,6,'0.7<=z<0.9',color='k')
CSb = axs.scatter(train_embedding[:, 0][split_b], train_embedding[:, 1][split_b], 0.01, c=age_train[split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split,vmax=np.max(age_train)))
cbarb = fig.colorbar(CSb)
cbara.set_label('age<3.0 Gyr')
cbarb.set_label('3.0 Gyr<=age')
#axs.set_xlim([-6.5,10.25])
#axs.set_ylim([-10.5,10.5])
plt.show()
"""
