# this is about generating a random data set and gen random minibatch from it as well
import numpy as np
import os
train_set=[];
test_set=[];

train_name='facebook/facebook.train.npy';
test_name='facebook/facebook.test.npy';
u_size=4039; # id from 0 - 4038

current_dir=os.path.dirname(os.path.abspath(__file__));
train_path=os.path.join(current_dir,train_name);
test_path=os.path.join(current_dir,test_name);




# once initialize, enough
def initialize():
    global train_set
    global test_set
    train_set=np.load(train_path);
    test_set=np.load(test_path);
    return;


#　it also includes the problem of how to generate fake negative samples
#　neg_ratio: param : the ratio of faked negative samples in a batch
#  only implement a simple mechanism here
#　 in future, it may be more complicated, or use an auxiliary net to do it
def gen_batch(batch_size,neg_ratio=0.5):
    fake_size=int(neg_ratio*batch_size);
    truth_size=batch_size-fake_size;
    fake_set=np.zeros((fake_size,3),dtype=np.int32);
    # first set the fake data in the batch
    fake_set[:,0]=np.random.choice(u_size,fake_size);
    fake_set[:,1]=np.random.choice(u_size,fake_size);
    # gen the truth set
    truth_set=np.ones((truth_size,3),dtype=np.int32);
    truth_set[:,0:2]=train_set[np.random.choice(train_set.shape[0],truth_size),:];
    #　gen batch
    batch=np.concatenate((fake_set,truth_set),axis=0);
    del truth_set
    del fake_set
    # random perm batch to get randomization benefits
    return batch[np.random.permutation(batch_size),:];


#　simple test for the correctness of this feeder 
if  __name__=='__main__':
    initialize();
    print(gen_batch(batch_size=100));
