# an implementation of the structure of the denoising autoencoder
# since denoising autoencoder is in the context of unsupervised learning
# which means that the loss function can also be given
import tensorflow as tf
import numpy as np
# configure the lib search path
import sys
sys.path.append('/home/morino/Documents/netize');


class DenoisingAE(object):
    """docstring for DenoisingAE.
        ::param input_layer  "access for other modules"
        ::param structure shape (3,1) [n_in, n_hidden, n_out] s.t. n_in==n_out
        ::param err_mode string "the metric of reconstruction error"
        ::param noise_mode string  "the kind of prior noise"
    """
    
    def __init__(self, input_layer, structure, err_mode='gaussian', noise_mode='gaussian'):
        self.arg = arg



#
