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


def gaussian_noise(input_layer,bias=0.0,sigma=0.1):
    noised_input=input_layer+tf.random_normal(tf.shape(input_layer),mean=bias,stddev=sigma);
    return noised_input;
    





# which is basically for binary-valued input as is suggested in [vincent10]
"""
::param fract [scalar] \in [0,1]  the proption of elements needed to be tampered
"""
def sault_and_pepper(input_layer,fract):
    tamper_num=tf.size(input_layer)*fract;
    # maybe a binary matrix form for choicing is better
    pass;
#
