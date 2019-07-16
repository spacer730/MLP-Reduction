import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1)) #-1 means we dont specify and let numpy figure out the correct dimension to make the other specified dimensions work
x = np.divide(x, 255.)

#We can do everything we did in the full tensorflow worked out program
#in a lot less lines of code using the tf.keras module.
t_model = tf.keras.Sequential()
t_model.add(tf.keras.layers.Dense(784, input_shape=(784,)))
t_model.add(tf.keras.layers.Dense(128, name='bottleneck'))
t_model.add(tf.keras.layers.Dense(784, activation=tf.nn.sigmoid))
t_model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss=tf.losses.sigmoid_cross_entropy)

#To get the feature representation of the network we do this:
session = tf.get_default_session()
if(session == None):
    session = tf.Session()

t_model.fit(x, x, batch_size=32, epochs=10)

#The following functions are to get the names for the specific layers/tensors, so that if we want
#to evaluate and save them for a certain input/output we can call them
# Get input tensor
def get_input_tensor_variable(model):
    return model.layers[0].input

# get bottleneck tensor
def get_bottleneck_tensor_variable(model):
    return model.get_layer(name='bottleneck').output

# Get output tensor
def get_output_tensor_variable(model):
    return model.layers[-1].output

t_input = get_input_tensor_variable(t_model)
t_enc = get_bottleneck_tensor_variable(t_model)
t_dec = get_output_tensor_variable(t_model)

# enc will store the actual encoded vanaalues of x[0:1]
enc = session.run(t_enc, feed_dict={t_input:x[0:1]})
# dec will store the actual decoded values of enc
dec = session.run(t_dec, feed_dict={t_enc:enc})
# reconstructed will store the reconstructed output of the input
reconstructed = session.run(t_dec, feed_dict={t_input:x[0:1]})


