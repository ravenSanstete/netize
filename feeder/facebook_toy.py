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
train_adj_graph=list();
test_adj_graph=list();
for i in range(u_size):
    train_adj_graph.append(list());
    test_adj_graph.append(list());




# once initialize, enough
def initialize():
    global train_set
    global test_set
    train_set=np.load(train_path);
    test_set=np.load(test_path);
    # to construct the adj graph of the test set
    for i in range(train_set.shape[0]):
        train_adj_graph[train_set[i][0]].append(train_set[i][1]);
    for i in range(test_set.shape[0]):
        test_adj_graph[test_set[i][0]].append(test_set[i][1]);
    return;


#　it also includes the problem of how to generate fake negative samples
#　neg_ratio: param : the ratio of faked negative samples in a batch
#  only implement a simple mechanism here
#　 in future, it may be more complicated, or use an auxiliary net to do it

# to mark the negative examples as -1, thus may use the cross entropy
def gen_batch(batch_size,neg_ratio=0.8):
    fake_size=int(neg_ratio*batch_size);
    truth_size=batch_size-fake_size;

    # to set the weight of the negative set to do some observation work
    fake_set=0*np.ones((fake_size,3),dtype=np.int32);
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

# return the precision of the link prediction model
# a standare criterion for this problem
def precision(w_mat,L):
    # normalize w_mat matrix
    w_mat-=np.tril(w_mat);
    flat_w=np.ndarray.flatten(w_mat);
    sorted_indices=np.argsort(flat_w);
    # then may need to do an iteration from the list end
    count=0;
    hit=0;
    train_hit=0;
    for i in range(sorted_indices.size):
        # termination condition, which means that already L test cases have been processed
        if(count>=L):
            break;
        if(i==0):
            cur_index=sorted_indices[-i-1:];
        else:
            cur_index=sorted_indices[-i-1:-i];
        pos_x=int(cur_index[0]/u_size);
        pos_y=int(cur_index[0]-pos_x*u_size);
        # thus there will be three cases
        if(pos_y in test_adj_graph[pos_x]):
            count+=1;
            hit+=1;
        elif(pos_y in train_adj_graph[pos_x]):
            train_hit+=1;
        else:
            count+=1;


    # top_L=sorted_indices[-L:];
    # # convert the indices in a tuple way
    # top_l_pos=np.zeros((L,2),dtype=np.int32);
    # # there is a need to store the temporary position info
    # for i in range(L):
    #     top_l_pos[i,0]=top_L[i]/u_size;
    #     top_l_pos[i,1]=top_L[i]-top_l_pos[i,0]*u_size;
    # # since the matrix is normalized into a lower triangle matrix, it's certainly that the row index is bigger than the column index
    # hit=0;
    # for i in range(L):
    #     if (top_l_pos[i,1] in test_adj_graph[top_l_pos[i,0]]):
    #         hit+=1;
    #     elif(top_l_pos[i,1])
    return hit/L,train_hit/L,;

#　simple test for the correctness of this feeder
if  __name__=='__main__':
    initialize();
    print(gen_batch(batch_size=100));
