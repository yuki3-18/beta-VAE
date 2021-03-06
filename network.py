'''
# Network Architecture 2 dim
# Author: Zhihui Lu
# Date: 2018/10/17
'''
import tensorflow as tf
import numpy as np

def cnn_encoder(input, latent_dim):
    # encoder layer
    encoder = cnv_2d(input, filters=64, kernel_size=(5,5), strides=(2,2), padding='same', name='encoder-01')
    encoder = cnv_2d(encoder, filters=128, kernel_size=(5,5), strides=(2,2) ,padding='same', name='encoder-02')
    encoder = cnv_2d(encoder, filters=256, kernel_size=(5,5), strides=(2,2) ,padding='same', name='encoder-03')
    encoder = cnv_2d(encoder, filters=512, kernel_size=(5,5), strides=(2,2) ,padding='same', name='encoder-04')
    encoder = cnv_2d(encoder, filters=768, kernel_size=(5,5), strides=(2,2) ,padding='same', name='encoder-final')

    shape_before_flatten = tuple(encoder.get_shape().as_list())

    # encoder to latent space
    output = tf.layers.flatten(encoder)
    output = tf.layers.dense(output, 2 * latent_dim, name='encoder-latent')

    return output, shape_before_flatten

def cnn_decoder(input, batch_size, shape_before_flatten):
    # latent space to decoder
    decoder = tf.layers.dense(input, np.prod(shape_before_flatten[1:]), activation=tf.nn.relu, name='decoder-latent')
    decoder = tf.reshape(decoder, [batch_size, shape_before_flatten[1], shape_before_flatten[2], shape_before_flatten[3]])

    # decoder layer
    decoder = decnv_2d(decoder, filters=512, kernel_size=(5,5), strides=(2,2), padding='same', name='decoder-01')
    decoder = decnv_2d(decoder, filters=256, kernel_size=(5,5), strides=(2,2), padding='same', name='decoder-02')
    decoder = decnv_2d(decoder, filters=128, kernel_size=(5,5), strides=(2,2), padding='same', name='decoder-03')
    decoder = decnv_2d(decoder, filters=64, kernel_size=(5,5), strides=(2,2), padding='same', name='decoder-04')
    output = tf.layers.conv2d_transpose(decoder, filters=1, kernel_size=(5,5), strides=(2,2), padding='same',
                                        activation=tf.tanh,name='decoder-final')
    return output


def cnv_2d(input, filters, name, kernel_size=(5,5), strides=(2,2), padding='same'):
    output = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)
    output = tf.layers.batch_normalization(output, axis=-1)
    output = tf.nn.relu(output)
    return output

def decnv_2d(input, filters, name, kernel_size=(5,5), strides=(2,2), padding='same'):
    output = tf.layers.conv2d_transpose(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)
    output = tf.nn.relu(output)
    return output


# # for mnist demo
def mnist_encoder(input, latent_dim):
    encoder = tf.layers.dense(input, 500, activation = tf.nn.relu, name='encoder_01')
    shape_size = tuple(encoder.get_shape().as_list())
    output = tf.layers.dense(encoder, 2 * latent_dim, name='encoder-latent')

    return output, shape_size

def mnist_decoder(input, batch_size, shape_size):
    decoder = tf.layers.dense(input, 500, activation = tf.nn.relu, name='decoder-01')
    output = tf.layers.dense(decoder, 784, activation = tf.nn.sigmoid)

    return output