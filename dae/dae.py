# to implement a denoising deep autoencoder with common usage

class DeepAutoEncoder(object):
    """docstring for DeepAutoEncoder."""
    def __init__(self, shape):
        self.shape=shape; # shape as [input, hidden_1, hidden_2, hidden_3, ...,output]
        self.sigmoid_layers=[];
        self.da_layers=[];
        self.hidden_layer_size=len(shape)-2;

        # init sigmoid_layer and da_layer instances


















#
