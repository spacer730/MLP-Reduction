import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import tsne

if __name__ == "__main__":
    np.random.seed(33)
    print('Openining the data file')
    fdata = fits.open('Data/COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012_removed_magnitudes_larger_90.fits')[1].data
    print('File is open')
    Ha = fdata['MAG_GAAPadapt_H']
    H = fdata['MAG_GAAP_H']
    Hflux = fdata['FLUX_GAAP_H']
    Ja = fdata['MAG_GAAPadapt_J']
    J = fdata['MAG_GAAP_J']
    Jflux = fdata['FLUX_GAAP_J']
    Ksa = fdata['MAG_GAAPadapt_Ks']
    Ks = fdata['MAG_GAAP_Ks']
    Ksflux = fdata['FLUX_GAAP_Ks']
    Ya = fdata['MAG_GAAPadapt_Y']
    Y = fdata['MAG_GAAP_Y']
    Yflux = fdata['FLUX_GAAP_Y']
    Za = fdata['MAG_GAAPadapt_Z']
    Z = fdata['MAG_GAAP_Z']
    Zflux = fdata['FLUX_GAAP_Z']
    ga = fdata['MAG_GAAPadapt_g']
    g = fdata['MAG_GAAP_g']
    gflux = fdata['FLUX_GAAP_g']
    ia = fdata['MAG_GAAPadapt_i']
    i = fdata['MAG_GAAP_i']
    iflux = fdata['FLUX_GAAP_i']
    ra = fdata['MAG_GAAPadapt_r']
    r = fdata['MAG_GAAP_r']
    rflux = fdata['FLUX_GAAP_r']
    ua = fdata['MAG_GAAPadapt_u']
    u = fdata['MAG_GAAP_u']
    uflux = fdata['FLUX_GAAP_u']
    zspec = fdata['zspec']

    X = np.array([H, J, Ks, Y, Z, g, i, r, u])
    X = X.transpose() #This ensures that each array entry is the 9 magnitudes of a galaxy

    #Shuffle data and build training and test set
    permuted_indices = np.random.permutation(len(X))
    X_perm = X[permuted_indices]
    zspec_perm = zspec[permuted_indices]

    cut_off = int(0.8*len(X_perm))
    X_train = X_perm[0:cut_off]
    X_test = X_perm[cut_off::]

    #Normalize data
    X_train = (X_train-np.min(X))/(np.max(X)-np.min(X))
    X_test = (X_test-np.min(X))/(np.max(X)-np.min(X))

    #tsne(data-array, reduced_dim, original_dim, perplexity)
    Sol = tsne.tsne(X_perm[0:1000], 2, 9, 30) #Two-dimensional coordinates from the t-sne algorithm from the first 1000 galaxies

    #Plot the results
    fig, axs = plt.subplots()
    CS = axs.scatter(Sol[:, 0], Sol[:, 1], 10, c=zspec_perm[0:1000], cmap='Blues') #x,y coordinates and the size of the dot
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('spectral redshift')
    plt.show()
