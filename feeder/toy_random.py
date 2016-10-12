# this is about generating a random data set and gen random minibatch from it as well
import numpy as np

data=[];

# once initialize, enough
def initialize(sample_num=1000,dim=10):
    global data;
    data=np.random.randn(sample_num,dim);
    return;

def gen_batch(batch_size):
    return data[np.random.choice(data.shape[0],batch_size),:];
