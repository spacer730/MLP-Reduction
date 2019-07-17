import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Load the data en reshape it in the form we want
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1)) #-1 means we dont specify and let numpy figure out the correct dimension to make the other specified dimensions work
x_train = np.divide(x_train, 255.)
x_test = x_test.reshape((x_test.shape[0], -1))
x_test = np.divide(x_test, 255.)

#Build the layers of the autoencoder and connect them in the correct way
input_layer = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(128, activation = 'relu')(input_layer)
encoded = tf.keras.layers.Dense(64, activation = 'relu')(encoded)
encoded = tf.keras.layers.Dense(32, activation = 'relu', name = 'bottleneck')(encoded)

decoded = tf.keras.layers.Dense(64, activation = 'relu')(encoded)
decoded = tf.keras.layers.Dense(128, activation = 'relu')(decoded)
decoded = tf.keras.layers.Dense(784, activation ='sigmoid')(decoded)

#this model maps it input onto a reconstruction of itself
autoencoder = tf.keras.models.Model(input_layer, decoded)

#this models maps its input onto its encoded representation
encoder = tf.keras.models.Model(input_layer, encoded) #model.get_layer(name='bottleneck'))

#to make a model for the decoder we have to make a new layer that acts as the encoded layer since the previous layer
#is linked to the input layer. We will use the decoded_layer from the autoencoder and link that to this encoded layer.
#this ensures that when the autoencoder gets trained, the weights associated with the decoded layer are the trained ones.
encoded_input = tf.keras.layers.Input(shape=(32,))
decoded_layer1 = autoencoder.layers[-3](encoded_input)
decoded_layer2 = autoencoder.layers[-2](decoded_layer1)
decoded_layer3 = autoencoder.layers[-1](decoded_layer2)
decoder = tf.keras.models.Model(encoded_input, decoded_layer3)

#lets train our autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

#after the model weights have been trained, we will do some tests to see how well the autoencoder works
reconstructed_imgs = autoencoder.predict(x_test)
encoded_codes = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_codes)

# now using Matplotlib to plot the images
n = 10 # how many images we will display
plt.figure(figsize=(20, 6))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i+50].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(reconstructed_imgs[i+50].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 1 + 2*n)
    
    plt.imshow(decoded_imgs[i+50].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
