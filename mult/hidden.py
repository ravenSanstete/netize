# a sigmoid layer component for other modules usage
import tensorflow as tf
import utils as utils
class HiddenLayer(object):
    """a two level neural structure, basic use as a component for more complicated structure
    # ::param input_layer [batch_size,in_dim]
    # ::param in_dim scalar
    # ::param out_dim scalar
    # ::param W [in_dim,out_dim] ::default None ::name "weight matrix"
    # ::param b [out_dim,1] ::default None ::name "bias"
    # ::param mode string ::default "sigmoid" ::name "nonlinearity mode"
    """

    def __init__(self, input_layer,in_dim,out_dim,W=None,b=None,mode='sigmoid'):
        self.input_layer=input_layer; # of size [batch_size,in_dim] shape only as one-dim
        self.in_dim=in_dim; # scalar
        self.out_dim=out_dim; # scalar
        if(W=None):
            pass; # init W
        if(b=None):
            pass; # init b
        self.W=W;
        self.b=b;
        self.activate_func=utils.nonlinearity(mode);
    # define the output of this layer
    def out():
        pass; # compute the output of this layer
    def variables():
        pass; # return the variables that needs to be initialized in a session
