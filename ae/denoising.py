# an implementation of the structure of the denoising autoencoder
# since denoising autoencoder is in the context of unsupervised learning
# which means that the loss function can also be given
# ぼくの哲学になか、training と　結構　は　ほんとにちがいものだ。
#　

import tensorflow as tf
import numpy as np
# configure the lib search path

import mult.hidden as weaver
import utils.interfaces as utils
# common interface: A(original_code, reconstruct_code,name)
# no default value, 命名は必要。
# [Note: original_code and reconstruct_code is of the same type]
# for real-valued code

# need to compute the average
def gaussian_re(src_code,reconstr_code,_name):
    return tf.reduce_mean(tf.pow(src_code-reconstr_code,2),name=_name);
# for binary valued x
def ce_re(src_code,reconstr_code,_name):
    return tf.reduce_mean(-(src_code*tf.log(reconstr_code)+(1-src_code)*tf.log(1-reconstr_code)),name=_name);

# define some reconstruction error functions right here and use a dictionary to index them
reconstruct_err_mode={
    "gaussian":gaussian_re,
    "cross_entropy":ce_re
}


# 選択は不要なことだ。
# 終了です
class DenoisingAE(object):
    """docstring for DenoisingAE.
        ::param input_layer  "access for other modules"
        ::param structure shape (2,1) [n_in, n_hidden, n_out] s.t. n_in==n_out
        ::param err_mode string "the metric of reconstruction error"
        ::param noise_mode string  "the kind of prior noise"
        ::param sample_num scalar 
        [Note: the reason for introducing the input_layer dimension into the structure shape is because that
        It's more convenient to use it directly, it's a quite common-use constant.
        ]
        そのネットの骨組みはほかの基本なもののうえにいきている。「基本じゃなくて、ならばいいです。」
    """
    # first let all the nonlinearity component to be of the form sigmoid, which makes it a little bit clearer

    # may need to add some options structure to spectify the noise parameter, いま、やらないでいい。
    def __init__(self, input_layer,sample_num, structure, err_mode='gaussian', noise_mode='gaussian',name="denoising_ae"):
        # do basic assignment
        self.components=[];
        self.name=name;
        self.sample_num=sample_num;
        self.structure=structure;
        self.noise_mode=noise_mode;
        self.err_mode=err_mode;
        # 実例化いろいろきほんなこと
        self.input_layer=input_layer; #　to restore the original input_layer value, for the calculation of the reconstruction error
        self.components.append(weaver.HiddenLayer(utils.noise(self.noise_mode)(input_layer),self.sample_num,structure[0],structure[1],_name=self.name+'_encoder'));
        # add tied weight 
        
        self.components.append(weaver.HiddenLayer(self.code_out(),self.sample_num,structure[1],structure[0],W=tf.transpose(self.components[0].W),_name=self.name+'_decoder'));
        # おしまい
    # to define the output of the network
    def code_out(self):
        return self.components[0].out(); #simply the  でりぐち of the second ネット
    def out(self):
        return self.components[1].out();
    # use such simple naming 、将来のために
    def loss(self):
        return reconstruct_err_mode[self.err_mode](self.input_layer,self.out(),_name=self.name+'_reconstruction_error');
    # return the variables of this model
    def variables(self):
        return [self.components[0].b,self.components[0].W,self.components[1].b]; # only includes all the non-repeated variables
        






#
