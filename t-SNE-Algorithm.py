import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import tsne

def luptitude(m0, flux):
    a = 2.5
    mu0 = m0 - 2.5*np.log(b)
    mu = mu0 - a*np.arcsinh(flux/(2*b))


if __name__ == "__main__":
    print('Openining the data file')
    fdata = fits.open('Data/COSMOSadaptdepth_ugriZYJHKs_rot_photoz_x_G10CosmosCatv04_plus_observed_targets_09October2012.fits')[1].data
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

    #Remove the galaxies that have a magnitude value of 99, which is likely an error
    maxes = np.array([np.max(X[i]) for i in range(len(X))]) #largest magnitude value of each galaxy
    goodindices = np.array(range(len(maxes)))[maxes<90] #indices of the galaxies that have a largest magnitude smaller than 90
    Xgood = X[goodindices] #magnitudes of the galaxies that have a largest magnitude less than 90
    zspec_good = zspec[goodindices]

    #tsne(data-array, reduced_dim, original_dim, perplexity)
    Sol = tsne.tsne(Xgood[0:1000], 2, 50, 5.0) #Two-dimensional coordinates from the t-sne algorithm from the first 1000 galaxies

    #Plot the results
    fig, axs = plt.subplots()
    CS = axs.scatter(Sol[:, 0], Sol[:, 1], 10, c=zspec_good[0:1000], cmap='Blues') #x,y coordinates and the size of the dot
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('spectral redshift')
    plt.show()
