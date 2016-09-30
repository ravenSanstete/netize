import tensorflow as tf;
import noise_maker

# dictionary for different usages
nonlinearity_dict={
    'sigmoid':tf.sigmoid,
    'tanh':tf.tanh
};

noise_dict={
    'gaussian':noise_maker.gaussian_noise,
    'mask':noise_maker.mask_noise,
    'sault':sault_and_pepper,
    'none':none_noise
}

# basically for hidden layer
def nonlinearity(name):
    return nonlinearity_dict[name];

# basically for denoising autoencoder
def noise(name):
    return noise_dict[name];
