import tensorflow as tf;
nonlinearity_dict={
    'sigmoid':tf.nn.sigmoid;
};
def nonlinearity(name):
    return nonlinearity_dict[name];

#
