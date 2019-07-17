import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tsne
from astropy.io import fits

#Load the data en reshape it in the form we want
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

#Build the layers of the autoencoder and connect them in the correct way
input_layer = tf.keras.layers.Input(shape=(9,))
encoded = tf.keras.layers.Dense(7, activation = 'relu')(input_layer)
encoded = tf.keras.layers.Dense(5, activation = 'relu')(encoded)
encoded = tf.keras.layers.Dense(2, activation = 'relu', name = 'bottleneck')(encoded)

decoded = tf.keras.layers.Dense(5, activation = 'relu')(encoded)
decoded = tf.keras.layers.Dense(7, activation = 'relu')(decoded)
decoded = tf.keras.layers.Dense(9, activation = 'sigmoid')(decoded)

#this model maps it input onto a reconstruction of itself
autoencoder = tf.keras.models.Model(input_layer, decoded)

#this models maps its input onto its encoded representation
encoder = tf.keras.models.Model(input_layer, encoded) #model.get_layer(name='bottleneck'))

#to make a model for the decoder we have to make a new layer that acts as the encoded layer since the previous layer
#is linked to the input layer. We will use the decoded_layer from the autoencoder and link that to this encoded layer.
#this ensures that when the autoencoder gets trained, the weights associated with the decoded layer are the trained ones.
encoded_input = tf.keras.layers.Input(shape=(2,))
decoded_layer1 = autoencoder.layers[-3](encoded_input)
decoded_layer2 = autoencoder.layers[-2](decoded_layer1)
decoded_layer3 = autoencoder.layers[-1](decoded_layer2)
decoder = tf.keras.models.Model(encoded_input, decoded_layer3)

#lets train our autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=100, batch_size=16, shuffle=True, validation_data=(X_test, X_test))

#after the model weights have been trained, we will do some tests to see how well the autoencoder works
reconstructed_mags = autoencoder.predict(X_test)
encoded_mags = encoder.predict(X_test)
decoded_mags = decoder.predict(encoded_mags)

np.linalg.norm(X_test[1]-reconstructed_mags[1])/np.linalg.norm(X_test[1])

#visualize the distribution of galaxies in the compressed feature space
fig, axs = plt.subplots()
CS = axs.scatter(encoded_mags[:, 0], encoded_mags[:, 1], 10, c=zspec_perm[cut_off::], cmap='Blues') #x,y coordinates and the size of the dot
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('spectral redshift')
plt.show()
