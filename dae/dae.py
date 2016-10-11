# to implement a denoising deep autoencoder with common usage
# pretrain procedure should not be implemented in this file
# since dae is just a structure without any idea of where it will be applied
import numpy as np
import tensorflow as tf

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import mult.hidden as hidden
import ae.denoising as denoising

class DeepAutoEncoder(object):
    """
    ::param input_layer  "access for other modules"
    ::param sample_num "the input data numbers"
    ::param shape "the whole structure parameters"
    """

    def __init__(self, input_layer, sample_num ,shape):
        self.shape=shape; # shape as [input, hidden_1, hidden_2, hidden_3, ...,output]
        self.sigmoid_layers=[];
        self.da_layers=[];
        self.hidden_layer_size=len(shape)-1;

        # should only initialize ae layers here and after training , store the ae parameters
        # for mlp's usage
        current_input=input_layer;
        for i in range(self.hidden_layer_size):
             ae_layer=denoising.DenoisingAE(current_input,sample_num,[self.shape[i],self.shape[i+1]]);
             # constrain the sigmoid layer weight matrix the same as the denoising structure
             # in a more general word, they are sharing the same structure
             sig_layer=hidden.HiddenLayer(current_input,sample_num,self.shape[i],self.shape[i+1],W=(ae_layer.variables()[1]));
             self.da_layers.append(ae_layer);
             current_input=ae_layer.code_out();
    # construct the loss functions
    # actually returns a loss list composed with the reconstruction losses of all the autoencoder layers
    def loss(self):
        loss_list=[];
        for i in range(self.hidden_layer_size):
            loss_list.append(self.da_layers[i].loss());
        return loss_list;
    # collect each layers output as a list
    def out(self):
        out_list=[];
        for i in range(self.hidden_layer_size):
            out_list.append(self.sig_layer.out());
        return out_list;
    def variables(self):
        var_list=[];
        for i in range(self.hidden_layer_size):
            var_list.extend(self.da_layers[i].variables());
        return var_list;
























#
