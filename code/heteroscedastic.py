import tensorflow as tf
from tensorflow import keras
import numpy as np

def gaussian_nll_loss(y_true, y_pred):
    mean = tf.expand_dims(y_pred[:,0], axis=1)
    log_variance = tf.expand_dims(y_pred[:,1], axis=1)

    # Calculate the negative log-likelihood
    mse = keras.losses.mean_squared_error(y_true, mean)
    variance = tf.exp(log_variance)
    pi = tf.constant(3.141592653589)
    nll = 0.5 * (tf.math.log(2 * pi * variance) + mse / variance)

    # Return the average NLL across the batch
    return tf.reduce_mean(nll)
    

def laplace_nll_loss(y_true, y_pred):
    mu = tf.expand_dims(y_pred[:,0], axis=1)
    log_b = tf.expand_dims(y_pred[:,1], axis=1)

    # Calculate the absolute error
    abs_error = tf.abs(y_true - mu)

    # Calculate the negative log-likelihood
    b = tf.exp(log_b)
    nll = abs_error / b + log_b + tf.math.log(2.0)

    # Return the average NLL across the batch
    return tf.reduce_mean(nll)


def cauchy_nll_loss(y_true, y_pred):
    mu = tf.expand_dims(y_pred[:,0], axis=1)
    log_b = tf.expand_dims(y_pred[:,1], axis=1)
    pi = tf.constant(3.141592653589)
    # Calculate the negative log-likelihood
    b = tf.exp(log_b)
    nll = tf.math.log(pi * b) + tf.math.log(1 + tf.square((y_true - mu) / b))

    # Return the average NLL across the batch
    return tf.reduce_mean(nll)


