import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, D, d):
        #This is a variable outside of the model that will be used to give the
        #specifications for the first layer of the model. It is a placeholder, because later on we will tell tensorflow
        #which data we will use to be put in here. Often (depending on the archetype of the network) it is the first layer.
        self.X = tf.placeholder(tf.float32, shape=(None, D))

        #The weights and biases' variables that convert the values of the nodes of the
        #first layer to the second layer. The weight values are taken from a normal
        #distribution of mean 0 and standard deviation 1. The bias values are zero
        self.W1 = tf.Variable(tf.random_normal(shape=(D,d)))
        self.b1 = tf.Variable(np.zeros(d).astype(np.float32))

        #The tensor that represents the bottleneck layer. We will use a relu activation
        #function that will be applied to the matrix product of the first layer
        #and the weights plus the biases
        self.Z = tf.nn.relu( tf.matmul(self.X, self.W1) + self.b1 )

        #Same stuff, but from the middle layer to the output layer.
        self.W2 = tf.Variable(tf.random_normal(shape=(d,D)))
        self.b2 = tf.Variable(np.zeros(D).astype(np.float32))

        #The activation function for the output layer has simply been chosen to be a sigmoid function,
        #since sigmoid output range is [0,1] (same range as the input value of the normed pixels)
        logits = tf.matmul(self.Z, self.W2) + self.b2
        self.X_hat = tf.nn.sigmoid( logits )

        #We choose the sigmoid cross entropy loss function since the values of the input vector are between [0,1]
        #The function returns a tensor of the same shape as logits with component wise logistic losses
        #reduce_sum here sums all the values of the tensor and returns one value: the loss/cost value
        self.loss = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits( labels=self.X, logits=logits ) )

        #The specific optimizer we use to train oour model uses the RMSProp algorithm (very good). Pretty much guesswork
        #when which optimizer performs the best.
        #The cost function defined above is what we want to minimize.
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.005).minimize(self.loss)

        #The follow code is used to connect with the backend and is needed to initialize the tensorflow graph.
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.get_default_session()
        if(self.sess == None):
            self.sess = tf.Session()
        self.sess.run(self.init_op)

    #To fit the model. Epochs is the number iterations over the entire dataset.
    #batch_size is the number of samples before the model parameters are updated.
    def fit(self, X, epochs=100, batch_size=64):
        #Number of batches is rounded down and in the case it is rounded down the last batch will be smaller than the others.
        n_batches = len(X) // batch_size

        for i in range(epochs):
            #Permute the input data
            X_perm = np.random.permutation(X)
            for j in range(n_batches):
                #Load data for current batch
                batch = X_perm[j*batch_size:(j+1)*batch_size]
                #Run the batch training!
                #Here we tell tensorflow to run the training step using batch as the 
                _, _ = self.sess.run((self.optimizer, self.loss), feed_dict={self.X: batch})

    #The following function predicts the network reconstructs given an input X
    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X: X})

    #Now that we are done training our network to reconstruct the input, we want to know the feature representation
    #of each of our data items:
    def encode(self, X):
        return self.sess.run(self.Z, feed_dict={self.X: X})

    #We can define something similar to decode a given feature to see what the reconstructed input vector is
    #according to our network
    def decode(self, Z):
        return self.sess.run(self.X_hat, feed_dict={self.Z: Z})

    #To close the tensorflow session
    def terminate(self):
        self.sess.close()
        del self.sess

def show_digit_image(dig):
    dig = dig*255.
    dig = dig.reshape((28,28))
    plt.imshow(dig)
    plt.show()

if __name__ == "__main__":
    mnist_autoencoder = Autoencoder(784, 128)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1)) #-1 means we dont specify and let numpy figure out the correct dimension to make the other specified dimensions work
    x = np.divide(x, 255.)
    mnist_autoencoder.fit(x)
    prediction = mnist_autoencoder.predict(x[0:1])
