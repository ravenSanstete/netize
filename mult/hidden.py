# a sigmoid layer component for other modules usage
import tensorflow as tf
import numpy as np


import utils.interfaces as utils



class HiddenLayer(object):
    """a two level neural structure, basic use as a component for more complicated structure
    # ::param input_layer [batch_size,in_dim]
    # ::param sample_num scalar \eq batch_size
    # ::param in_dim scalar
    # ::param out_dim scalar
    # ::param W [in_dim,out_dim] ::default None ::name "weight matrix"
    # ::param b [out_dim,1] ::default None ::name "bias"
    # ::param mode string ::default "sigmoid" ::name "nonlinearity mode"
    # ::param name string
    """

    def __init__(self, input_layer,sample_num,in_dim,out_dim,W=None,b=None,mode='sigmoid',_name='hidden_layer'):
        self.input_layer=input_layer; # of size [batch_size,in_dim] shape only as one-dim
        self.in_dim=in_dim; # scalar
        self.out_dim=out_dim; # scalar
        self.name=_name;
        self.sample_num=sample_num;
        # use the init value the same as the deeplearning tutorial(maybe it's more proper)
        _low=-6.0/(in_dim+out_dim);
        _high=6.0/(in_dim+out_dim);
        # sample W's entry from a unfirom
        if(W==None):
            W=tf.Variable(np.random.uniform(low=_low,high=_high,size=(in_dim,out_dim)),dtype=tf.float32,name=self.name+'_weight');
        # set b initial value to be zero
        if(b==None):
            b=tf.Variable(np.zeros((out_dim,1)),dtype=tf.float32,name=self.name+'_bias');
        self.W=W;
        self.b=b;
        self.activate_func=utils.nonlinearity(mode);
    # define the output of this layer
    def out(self):
        expanded_b=tf.matmul(tf.ones([self.sample_num,1]),self.b,transpose_b=True);
        return self.activate_func(tf.matmul(self.input_layer,self.W)+expanded_b,name=self.name+'_nl_trans');
    def variables(self):
        return (self.W,self.b); # return the variables that needs to be initialized in a session
