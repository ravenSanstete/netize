# methods for adding slight noise to the raw input_layer
# interface for methods in this module
# Func(input_layer, [options, ...] )
# ::return noised_input the same size as the input layer

import tensorflow as tf
import numpy as np

"""
::param bias [scalar] the gaussian's mean
::param sigma [scalar] the magnitude of the gaussian noise
[Note] never do assignment or something destructive to the raw input layer
"""
# please adding no noises, an identity transformation
def none_noise(input_layer):
    return input_layer;



def gaussian_noise(input_layer,bias=0.0,sigma=0.1):
    noised_input=input_layer+tf.random_normal(tf.shape(input_layer),mean=bias,stddev=sigma);
    return noised_input;


def mask_noise(input_layer,fract=0.1):
    # maybe a binary matrix form for choicing is better
    p=1-fract/tf.size(input_layer);
    projector=np.random.binomial(1,p,size=tf.shape(input_layer));
    return projector*input_layer; # do some projection according to the binary matrix


# a dictionary for padding value choosing
spice_mode={
    'min':tf.reduce_min,
    'max':tf.reduce_max,
    'mean':tf.reduce_mean,
}

# which is basically for binary-valued input as is suggested in [vincent10]
"""
::param fract [scalar] \in [0,1]  the proption of elements needed to be tampered
::param mode string choice of min val fill or max val fill
"""


# which implements the third method declared in vincent10's work
def sault_and_pepper(input_layer,fract=0.1,mode='min'):
    p=1-fract/tf.size(input_layer);
    padding_val=spice_mode[mode](input_layer); # declare
    projector=np.random.binomial(1,p,size=tf.shape(input_layer));
    outcome=projector*input_layer;
    projector=(numpy.ones(shape=tf.shape(input_layer))-projector); # 0-1 flip
    outcome+=padding_val*projector;
    return outcome;

#
