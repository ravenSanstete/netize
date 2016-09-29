import tensorflow as tf;
nonlinearity_dict={
    'sigmoid':tf.sigmoid,
    'tanh':tf.tanh
};
def nonlinearity(name):
    return nonlinearity_dict[name];

#
