import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
import matplotlib.colors
import matplotlib as mpl
from astropy.io import fits
from astropy.table import Table
from matplotlib import cm
from mayavi import mlab
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import copy

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR

def load_COSMOS2015_data(remove_problematic_galaxies=True):
    #Load the data and reshape it in the form we want
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
    if remove_problematic_galaxies == True:
        good_indices = np.argwhere((np.abs(zphoto)<90) & (np.abs(mass)<90) & (np.abs(SSFR)<90) & (age>100) & (source_type == 0)).flatten() #& (np.abs(X[:,-1])<=90) & (np.abs(X[:,-2])<=90)).flatten()
        X, zphoto, mass, SSFR, age= X[good_indices], zphoto[good_indices], mass[good_indices], SSFR[good_indices], age[good_indices]
    return X, zphoto, mass, SSFR, age, source_type

def load_ZCOSMOS_data():
    print('Openining the data files')
    fdata_1 = fits.open('Data/C3R2_x_COSMOS2015_no_doubles_old_ZCosmos_plus_CSC2.fits')[1].data
    fdata_2 = fits.open('Data/OldZCOSMOS_x_COSMOS2015_no_doubles_C3R2_plus_CSC2.fits')[1].data
    fdata_3 = fits.open('Data/C3R2_x_COSMOS2015_doubles_old_ZCosmos_plus_CSC2.fits')[1].data
    print('File is open')
    #
    u, B, V, r, ip, zpp = fdata_1['u_MAG_AUTO'], fdata_1['B_MAG_AUTO'], fdata_1['V_MAG_AUTO'], fdata_1['r_MAG_AUTO'], fdata_1['ip_MAG_AUTO'], fdata_1['zpp_MAG_AUTO']
    Y, J, H, Ks = fdata_1['Y_MAG_AUTO'], fdata_1['J_MAG_AUTO'], fdata_1['H_MAG_AUTO'], fdata_1['Ks_MAG_AUTO']
    ch1, ch2, ch3, ch4 = fdata_1['SPLASH_1_MAG'], fdata_1['SPLASH_2_MAG'], fdata_1['SPLASH_3_MAG'], fdata_1['SPLASH_4_MAG']
    #
    zphoto, mass, SSFR, age = fdata_1['PHOTOZ'], fdata_1['MASS_MED'], fdata_1['SSFR_MED'], fdata_1['AGE']
    source_type = fdata_1['TYPE']
    #
    Z_C3R2 = fdata_1['Redshift']
    #
    Xray_data1_indices = np.argwhere(np.array([len(fdata_1['name'][i]) for i in range(len(fdata_1))])>0).flatten()
    #
    u, B, V = np.append(u,fdata_2['u_MAG_AUTO']), np.append(B,fdata_2['B_MAG_AUTO']), np.append(V,fdata_2['V_MAG_AUTO'])
    r, ip, zpp = np.append(r,fdata_2['r_MAG_AUTO']), np.append(ip,fdata_2['ip_MAG_AUTO']), np.append(zpp,fdata_2['zpp_MAG_AUTO'])
    Y, J, H, Ks = np.append(Y,fdata_2['Y_MAG_AUTO']), np.append(J,fdata_2['J_MAG_AUTO']), np.append(H,fdata_2['H_MAG_AUTO']), np.append(Ks,fdata_2['Ks_MAG_AUTO'])
    ch1, ch2, ch3, ch4 = np.append(ch1,fdata_2['SPLASH_1_MAG']), np.append(ch2,fdata_2['SPLASH_2_MAG']), np.append(ch3,fdata_2['SPLASH_3_MAG']), np.append(ch4,fdata_2['SPLASH_4_MAG'])
    #
    zphoto, mass, SSFR, age = np.append(zphoto,fdata_2['PHOTOZ']), np.append(mass,fdata_2['MASS_MED']), np.append(SSFR,fdata_2['SSFR_MED']), np.append(age,fdata_2['AGE'])
    source_type = np.append(source_type,fdata_2['TYPE'])
    #
    Z_OldZCOSMOS = fdata_2['zspec']
    Instr_OldZCOSMOS = fdata_2['Instr']
    #
    Xray_data2_indices = np.argwhere(np.array([len(fdata_2['name'][i]) for i in range(len(fdata_2))])>0).flatten()
    #
    u, B, V = np.append(u,fdata_3['u_MAG_AUTO_1']), np.append(B,fdata_3['B_MAG_AUTO_1']), np.append(V,fdata_3['V_MAG_AUTO_1'])
    r, ip, zpp = np.append(r,fdata_3['r_MAG_AUTO_1']), np.append(ip,fdata_3['ip_MAG_AUTO_1']), np.append(zpp,fdata_3['zpp_MAG_AUTO_1'])
    Y, J, H, Ks = np.append(Y,fdata_3['Y_MAG_AUTO_1']), np.append(J,fdata_3['J_MAG_AUTO_1']), np.append(H,fdata_3['H_MAG_AUTO_1']), np.append(Ks,fdata_3['Ks_MAG_AUTO_1'])
    ch1, ch2, ch3, ch4 = np.append(ch1,fdata_3['SPLASH_1_MAG_1']), np.append(ch2,fdata_3['SPLASH_2_MAG_1']), np.append(ch3,fdata_3['SPLASH_3_MAG_1']), np.append(ch4,fdata_3['SPLASH_4_MAG_1'])
    #
    zphoto, mass, SSFR, age = np.append(zphoto,fdata_3['PHOTOZ_1']), np.append(mass,fdata_3['MASS_MED_1']), np.append(SSFR,fdata_3['SSFR_MED_1']), np.append(age,fdata_3['AGE_1'])
    source_type = np.append(source_type,fdata_3['TYPE_1'])
    #
    Z_C3R2_doubles = fdata_3['Redshift']
    Z_OldZCOSMOS_doubles = fdata_3['zspec']
    Instr_OldZCOSMOS_doubles = fdata_3['Instr']
    #
    Xray_data3_indices = np.argwhere(np.array([len(fdata_3['name_1'][i]) for i in range(len(fdata_3))])>0).flatten()
    #
    CSC_Xray_indices = np.append(Xray_data1_indices, Xray_data2_indices+len(fdata_1))
    CSC_Xray_indices = np.append(CSC_Xray_indices, Xray_data3_indices+len(fdata_1)+len(fdata_2))
    #
    COSMOS2015_Xray_indices = np.argwhere(source_type==2).flatten()
    union_indices = np.union1d(COSMOS2015_Xray_indices, CSC_Xray_indices)
    intersect_indices = np.intersect1d(COSMOS2015_Xray_indices, CSC_Xray_indices)
    #
    Z_spec = np.append(Z_C3R2, Z_OldZCOSMOS)
    Z_spec = np.append(Z_spec, Z_C3R2_doubles)
    #
    u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks = u-B, B-V, V-r, r-ip, ip-zpp, zpp-Y, Y-J, J-H, H-Ks
    #
    X = np.array([u, u_B, B_V, V_r, r_ip, ip_zpp, zpp_Y, Y_J, J_H, H_Ks])#, ch1, ch2])#, ch3, ch4]) ch3 and ch4 have many errors
    X = X.transpose() #This ensures that each array entry are the colors/magnitudes of a galaxy
    #
    return X, Z_spec, CSC_Xray_indices, COSMOS2015_Xray_indices

def load_Synthetic_data(size):
    sdata = Table.read('Data/Synthethic-galaxy-data-'+str(size)+'-samples.dat', format='ascii')
    sdata = np.array(sdata)
    #
    u, g, r, i, Z = sdata['u'], sdata['g'], sdata['r'], sdata['i'], sdata['z']
    u_noerr, g_noerr, r_noerr, i_noerr, Z_noerr = sdata['u_noerr'], sdata['g_noerr'], sdata['r_noerr'], sdata['i_noerr'], sdata['z_noerr']
    z_redshift = sdata['z_redshift']
    galaxy_type = sdata['type']
    #
    u_g, g_r, r_i, i_Z = u-g, g-r, r-i, i-Z
    u_g_noerr, g_r_noerr, r_i_noerr, i_Z_noerr = u_noerr-g_noerr, g_noerr-r_noerr, r_noerr-i_noerr, i_noerr-Z_noerr
    #
    u_i, g_i, Z_i = u-i, g-i, Z-i
    u_i_noerr, g_i_noerr, Z_i_noerr = u_noerr-i_noerr, g_noerr-i_noerr, Z_noerr-i_noerr
    #
    X = np.array([i, u_i, g_i, r_i, Z_i])
    X = X.transpose()
    return X, z_redshift, galaxy_type

def bin_data(X, zphoto, mass, SSFR, age, source_type, embedding, zphoto_limits):
    binned_indices = np.argwhere((zphoto>=zphoto_limits[0]) & (zphoto<zphoto_limits[1])).flatten()
    X_bin, zphoto_bin, mass_bin, SSFR_bin, age_bin = X[binned_indices], zphoto[binned_indices], mass[binned_indices], SSFR[binned_indices], age[binned_indices]
    source_type_bin, embedding_bin = source_type[binned_indices], embedding[binned_indices]
    return X_bin, zphoto_bin, mass_bin, SSFR_bin, age_bin, source_type_bin, embedding_bin

def mayavi_plot_3d(embedding, size, colors):
    figure = mlab.figure('myfig')
    figure.scene.disable_render = True
    nodes = mlab.points3d(embedding[:,0], embedding[:,1], embedding[:,2], scale_factor = size, figure = figure)
    mlab.axes(xlabel='x', ylabel='y', zlabel='z', nb_labels = 5, figure = figure)
    nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.mlab_source.dataset.point_data.scalars = colors
    figure.scene.disable_render = False
    mlab.show()

def mpl_plot_3d(embedding, size, colors, split):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    split_a = np.argwhere(colors<split).flatten()
    split_b = np.argwhere(colors>=split).flatten()
    #
    my_cmap_1 = copy.copy(plt.cm.summer)
    my_cmap_1.set_bad(my_cmap_1(0))
    #
    CSa = ax.scatter(embedding[:,0][split_a], embedding[:,1][split_a], embedding[:,2][split_a], s=size, c=colors[split_a], cmap=my_cmap_1, norm=matplotlib.colors.LogNorm(vmin=0.01,vmax=split))
    cbara = fig.colorbar(CSa)
    CSb = ax.scatter(embedding[:,0][split_b], embedding[:,1][split_b], embedding[:,2][split_b], s=size, c=colors[split_b], cmap='autumn_r', norm=matplotlib.colors.LogNorm(vmin=split, vmax=np.max(colors)))
    cbarb = fig.colorbar(CSb)
    cbara.set_label('Z < '+str(split))
    cbarb.set_label(str(split)+' <= Z')
    #
    ax.xaxis.pane.fill, ax.yaxis.pane.fill, ax.zaxis.pane.fill = False, False, False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.view_init(elev=0., azim=0) #Y-Z
    #ax.view_init(elev=0, azim=270) #X-Z
    ax.view_init(elev=90, azim=-90) #X-Y
    plt.show()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def knn_regressor(train_embedding, train_variable, test_embedding, test_variable):
    neigh = [KNeighborsRegressor(n_neighbors=i, weights='distance').fit(train_embedding, train_variable) for i in range(1,50)]
    test_variable_pred = [neigh[i].predict(test_embedding) for i in range(len(neigh))]
    score_pred = [neigh[i].score(test_embedding, test_variable) for i in range(len(neigh))]
    rmse_pred = [rmse(test_variable_pred[i], test_variable) for i in range(len(test_variable_pred))]
    return test_variable_pred, score_pred, rmse_pred

def SVR_regressor(train_embedding, train_variable, test_embedding, test_variable):
    svr1 = SVR().fit(train_embedding, train_variable)
    test_variable_pred = svr1.predict(test_embedding)
    score_pred = svr1.score(test_embedding, test_variable)
    rmse_pred = rmse(test_variable_pred, test_variable)
    return test_variable_pred, score_pred, rmse_pred

def hexbin_scatter_plot(mass_test, mass_test_pred):
    #Hexbin scatter plot of M_pred vs M_true (SED-fit)
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1], hspace=0.03) 

    ax0 = plt.subplot(gs[0])
    hb = ax0.hexbin(mass_test,mass_test_pred,vmin=1, gridsize=100, bins='log', cmap='plasma')
    cb = fig.colorbar(hb, ax=ax0)
    cb.set_label('N')
    ax0.set_ylabel(r'$\rm Log\ (M_{*}/M_{\odot}) \ knn-10 \ Input \ Space$',fontsize=20)
    ax0.plot([4,12],[4,12],'k--',linewidth=1)
    ax0.set_xlim([7,np.max(mass_test)])
    ax0.set_ylim([7,np.max(mass_test)])
    ax0.text(10.2,7.7,r'$\rm 0.7<z<0.9$',size=15,color='k')
    ax0.axes.get_xaxis().set_ticks([])
    plt.tick_params(axis='both', which='major', labelsize=13)

    ax2 = plt.subplot(gs[1])
    ax2.set_xlabel(r'$\rm Log\ (M_{*}/M_{\odot})\ COSMOS\ SEDfit$',fontsize=20, labelpad=10)
    ax2.set_ylabel(r'$\rm \Delta \ Log(M)$',fontsize=20,labelpad=10)
    ax2.set_ylim([-0.5,0.5])
    hb=ax2.hexbin(mass_test,mass_test-mass_test_pred,vmin=1, gridsize=100, bins='log', cmap='plasma')
    cb = fig.colorbar(hb, ax=ax2)
    cb.set_label('N')
    ax2.plot([6,12],[0,0],'k--')
    ax2.set_xlim([7,np.max(mass_test)])
    ax2.set_ylim([-1.,1.])
    ax2.yaxis.set_ticks([-1.0,-0.5,0.,0.5,1.0])
    plt.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    plt.show()

#Functions to find outliers in OldZCosmos data
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

if __name__ == "__main__":
    np.random.seed(33)

    #ZCosmos
    #X, Z_spec, CSC_Xray_indices, COSMOS2015_Xray_indices = load_ZCOSMOS_data()
    #X_train, X_test, Z_spec_train, Z_spec_test = train_test_split(X, Z_spec)

    #COSMOS2015
    X, zphoto, mass, SSFR, age, source_type = load_COSMOS2015_data()
    X_train, X_test, zphoto_train, zphoto_test, mass_train, mass_test, SSFR_train, SSFR_test, age_train, age_test = train_test_split(X, zphoto, mass, SSFR, age)

    #Synthetic dataset
    #X, z_redshift, galaxy_type = load_Synthetic_data(50000)
    #X_train, X_test, z_redshift_train, z_redshift_test, galaxy_type_train, galaxy_type_test = train_test_split(X, z_redshift, galaxy_type)

    #Train an instance of the UMAP algorithm class on X_train and compute positions for X_test
    trans = umap.UMAP(n_components=3).fit(X_train)
    train_embedding = trans.embedding_
    test_embedding = trans.transform(X_test)

    #mayavi_plot_3d(train_embedding, size=0.1, colors=zphoto_train)
    #mpl_plot_3d(train_embedding, size=1, colors=z_redshift, split=1.46)
    
    """
    #Here we call the functions to find the outliers in the OldZCOSMOS data
    indices_top_arr_redshift_deviation = outliers(0.6, 1, 1, top_percent=True, data_return=True)
    indices_top_21_redshift_deviation = indices_top_arr_redshift_deviation[1]
    indices_top_211_redshift_deviation = indices_top_arr_redshift_deviation[0]
    indices_top_21_weighted_redshift_deviation = outliers(1.8, 1, 1, top_percent=True, data_return=True)[3]
    indices_top_211_weighted_redshift_deviation = outliers(3.2, 1, 1, top_percent=True, data_return=True)[2]
    """

    #Here we call the functions to do the regression on the different galactic variables of the COSMOS2015 data
    mass_test_pred_knn, mass_pred_knn_score, mass_pred_knn_rmse = knn_regressor(train_embedding, mass_train, test_embedding, mass_test)
    mass_test_pred_SVR, mass_pred_SVR_score, mass_pred_SVR_rmse = SVR_regressor(train_embedding, mass_train, test_embedding, mass_test)

    mass_test_pred_knn_og, mass_pred_knn_score_og, mass_pred_knn_rmse_og = knn_regressor(X_train, mass_train, X_test, mass_test)
    mass_test_pred_SVR_og, mass_pred_SVR_score_og, mass_pred_SVR_rmse_og = SVR_regressor(X_train, mass_train, X_test, mass_test)

    #Training knn regressor for redshift on both the embedded data as well as the original data

    #hexbin_scatter_plot(mass_test, mass_test_pred_SVR)
    
    plt.plot([i+1 for i in range(len(mass_pred_knn_rmse))], mass_pred_knn_rmse)
    plt.xlabel('# nearest neighbors')
    plt.ylabel('rmse')
    plt.text(80, 0.060, 'mass')
    plt.show()

    #X_train_bin, zphoto_train_bin, mass_train_bin, SSFR_train_bin, age_train_bin, source_type_train_bin, train_embedding_bin = bin_data(X_train,zphoto_train,mass_train,SSFR_train,age_train,source_type_train,train_embedding,[0.7,0.9])
    #X_test_bin, zphoto_test_bin, mass_test_bin, SSFR_test_bin, age_test_bin, source_type_test_bin, test_embedding_bin = bin_data(X_test,zphoto_test,mass_test,SSFR_test,age_test,source_type_test,test_embedding)
