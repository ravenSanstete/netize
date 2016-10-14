# a feeder for the movie lens 100k data set

import zipfile
import numpy as np
PATH='ml-100k.zip';
FILE_NAME='ml-100k/u.data';


# unzipped the file and then read in the u.data
def read_data(path=PATH,file_name=FILE_NAME):
    print('unzip dataset');
    with zipfile.ZipFile(path) as ml100k_zip:
        with ml100k_zip.open(file_name) as ml100k:
            line=ml100k.read().decode('utf-8');
            raw=line.replace('\n','\t').split('\t');
    return raw[:-1];

# raw=read_data();


     
# to construct it as a tensor data set
# n%4 : 0 uid, 1 iid, 2 rating, 3 timestamp
def build_dataset(raw,modu=4):
    print('build dataset');
    assert len(raw)%modu==0;
    
    ds_size=len(raw)//modu;
    data=np.zeros(shape=(ds_size,modu),dtype=np.int32); # define a custom tensor;
    data_count=0;
    row_count=0;
    for field in raw:
        data[row_count,data_count%modu]=field;
        data_count+=1;
        if(data_count%modu==0):
            row_count+=1;
    return data

# 
# data=build_dataset(raw);  

    
    

# return the batch_uid,batch_iid,batch_labels
def gen_batch(data,batch_size):
    data_size=data.shape[0];
    composed_batch=data[np.random.choice(data_size,batch_size),:];
    return composed_batch;

def gen_order_batch(data,beg,batch_size):
    data_size=data.shape[0];
    composed_batch=data[beg:beg+batch_size,:];
    batch_uid=np.ndarray(shape=(batch_size),dtype=np.int32);
    batch_iid=np.ndarray(shape=(batch_size),dtype=np.int32);
    batch_label=np.ndarray(shape=(batch_size),dtype=np.int32);
    count=0;
    for field in composed_batch:
        batch_uid[count]=field[0]-1;
        batch_iid[count]=field[1]-1;
        batch_label[count]=field[2];
        count+=1;
    return batch_uid,batch_iid,batch_label;

def gen_split_batch(data,batch_size):
    data_size=data.shape[0];
    composed_batch=data[np.random.choice(data_size,batch_size),:];
    batch_uid=np.ndarray(shape=(batch_size),dtype=np.int32);
    batch_iid=np.ndarray(shape=(batch_size),dtype=np.int32);
    batch_label=np.ndarray(shape=(batch_size),dtype=np.int32);
    count=0;
    for field in composed_batch:
        batch_uid[count]=field[0]-1;
        batch_iid[count]=field[1]-1;
        batch_label[count]=field[2];
        count+=1;
    return batch_uid,batch_iid,batch_label;
# # try to generate a sample batch
# batch_uid,batch_iid,batch_label=gen_batch(data,batch_size=8);
# for i in range(8):
#     print('(%d,%d,%d)' %(batch_uid[i],batch_iid[i],batch_label[i]));





def gen_mat(data_set,u_len,i_len):
    mat=np.zeros((u_len,i_len),dtype=np.float32);
    ind=np.zeros((u_len,i_len),dtype=np.int32);
    for i in range(data_set.shape[0]):
        row=data_set[i];
        mat[row[0]-1][row[1]-1]=row[2];
        ind[row[0]-1][row[1]-1]=1;
    return [mat,ind];























#
