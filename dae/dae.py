# to implement a denoising deep autoencoder with common usage
import numpy as np
import tensorflow as tf
import ae.denoising as denoising

class DeepAutoEncoder(object):
    """
    ::param input_layer  "access for other modules"
    ::param sample_num "the input data numbers"
    ::param shape "the whole structure parameters"
    """
    
    def __init__(self, input_layer, sample_num ,shape):
        self.shape=shape; # shape as [input, hidden_1, hidden_2, hidden_3, ...,output]
        self.da_layers=[];
        self.hidden_layer_size=len(shape)-2;

        # should only initialize ae layers here and after training , store the ae parameters
        # for mlp's usage
        current_input=input_layer;
         for i in range(self.hidden_layer_size):
             ae_layer=denoising.DenoisingAE(current_input,sample_num,[self.shape[i],self.shape[i+1]]);
             self.da_layers.append(ae_layer);
             current_input=ae_layer.code_out();
    # construct the loss functions
    # actually returns a loss list composed with the reconstruction losses of all the autoencoder layers
    def loss(self):
        loss_list=[];
        for i in range(self.hidden_layer_size):
            loss_list.append(self.da_layers[i].loss());
        return loss_list;
    def variables(self):
        var_list=[];
        for i in range(self.hidden_layer_size):
            var_list.extend(self.da_layers[i].variables());
        return var_list;
            
    
        


















#
