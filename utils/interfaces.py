import tensorflow as tf;

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import utils.noise_maker as noise_maker

# dictionary for different usages
nonlinearity_dict={
    'sigmoid':tf.sigmoid,
    'tanh':tf.tanh
};

noise_dict={
    'gaussian':noise_maker.gaussian_noise,
    'mask':noise_maker.mask_noise,
    'sault':noise_maker.sault_and_pepper,
    'none':noise_maker.none_noise
}

# basically for hidden layer
def nonlinearity(name):
    return nonlinearity_dict[name];

# basically for denoising autoencoder
def noise(name):
    return noise_dict[name];
