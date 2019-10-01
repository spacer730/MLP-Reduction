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
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from pylab import savefig
from mpl_toolkits.mplot3d import Axes3D

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

#X_train, X_test, Z_spec_train, Z_spec_test = train_test_split(X, Z_spec)

trans = umap.UMAP(n_components=2).fit(X)
embedding = trans.embedding_

#Using isolationforest to find outliers based on on the original input space positions of the data (magnitude + colors)
#Could also try to add Z_spec to the original input space as an extra dimension to take out isolated points in that dimension.
Z_spec[2068]=0.001
Z_spec[2090]=0.001
clf = IsolationForest(n_estimators=600, max_samples=21114, contamination=0.01)
clf.fit(np.column_stack((embedding,np.log10(Z_spec))))

y_pred = clf.predict(np.column_stack((embedding,np.log10(Z_spec))))
pred_outlier_indices = np.argwhere(y_pred==-1).flatten()
intersect_indices_pred = np.intersect1d(CSC_Xray_indices, pred_outlier_indices)
len(pred_outlier_indices)
len(intersect_indices_pred)

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
CSa = axs.scatter(embedding[:, 0][split_a], embedding[:, 1][split_a], s=[30 if np.any(i==pred_outlier_indices) else 1 for i in split_a], c=Z_spec[split_a], cmap=my_cmap_1, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=split))
cbara = fig.colorbar(CSa)
CSb = axs.scatter(embedding[:, 0][split_b], embedding[:, 1][split_b], s=[30 if np.any(i==pred_outlier_indices) else 1 for i in split_b], c=Z_spec[split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(Z_spec)))
cbarb = fig.colorbar(CSb)
#axs.text(2.5,8, 'Source_type == 0, outliers')
cbara.set_label('Z_spec < 1.47')
cbarb.set_label('1.47 <= Z_spec')
#axs.set_xlim([-9.75,8.8])
#axs.set_ylim([-10.4,8.7])
plt.show()


#3D plot with UMAP 2D projection as XY base and Z_spec as third dimension
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

split = 1.47#np.mean(Z_spec)+np.var(Z_spec)**0.5
split_a = np.argwhere(Z_spec<split).flatten()
split_b = np.argwhere(Z_spec>=split).flatten()

my_cmap_1 = copy.copy(plt.cm.summer)
my_cmap_1.set_bad(my_cmap_1(0))

CSa = ax.scatter(embedding[:,0][split_a], embedding[:,1][split_a], np.log10(Z_spec)[split_a], s=[30 if np.any(i==pred_outlier_indices) else 1 for i in split_a], c=Z_spec[split_a], cmap=my_cmap_1, norm=matplotlib.colors.LogNorm(vmin=0.01,vmax=split))
cbara = fig.colorbar(CSa)
CSb = ax.scatter(embedding[:,0][split_b], embedding[:,1][split_b], np.log10(Z_spec)[split_b], s=[30 if np.any(i==pred_outlier_indices) else 1 for i in split_a], c=Z_spec[split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(Z_spec)))
cbarb = fig.colorbar(CSb)
cbara.set_label('Z_spec < '+str(split))
cbarb.set_label(str(split)+' <= Z_spec')

ax.xaxis.pane.fill, ax.yaxis.pane.fill, ax.zaxis.pane.fill = False, False, False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
#ax.set_zscale('log')
ax.grid(False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=0., azim=0) #Y-Z
#ax.view_init(elev=0, azim=270) #X-Z
#ax.view_init(elev=90, azim=-90) #X-Y
plt.show()
