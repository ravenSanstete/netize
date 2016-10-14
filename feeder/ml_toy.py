# a feeder for the movie lens 100k data set
# to implement the feeder for each 
import zipfile
import os
import numpy as np

PATH='ml-1m.zip';
train_path='ml-1m/train.dat';
test_path='ml-1m/test.dat'
zip_name='ml-1m.zip';


# add some configuration dict
# format str:list[zip_fname,train_fname,test_fname,u_size,i_size,delim]
config={
    '1m': {
        'zip_name':'movie_lens/ml-1m.zip',
        'train_name': 'ml-1m/train.dat',
        'test_name': 'ml-1m/test.dat',
        'u_size': 6040,
        'i_size': 3952,
        'delim':'::'
     },
   '100k':{
        'zip_name':'movie_lens/ml-100k.zip',
        'train_name': 'ml-100k/ua.base',
        'test_name': 'ml-100k/ua.test',
        'u_size': 943,
        'i_size': 1682,
        'delim': '\t'
     }
}


current_dir=os.path.dirname(os.path.abspath(__file__));






train_set=[];
test_set=[];



# version could be '1m', '100k', '10m'
def initialize(version='1m'):
    global train_set;
    global test_set;
    train_set=read_data(config[version]['zip_name'],config[version]['train_name'],config[version]['delim']);
    test_set=read_data(config[version]['zip_name'],config[version]['test_name'],config[version]['delim']);
    pass;


# unzipped the file and then read in the u.data
def read_data(zip_path,filename,delimiter):
    print('unzip dataset');
    with zipfile.ZipFile(path) as ml_zip:
        with ml_zip.open(file_name) as ml:
            line=ml.read().decode('utf-8');
            raw=line.replace(delimiter,'\t').replace('\n','\t').split('\t');
    return raw;



     
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
# 
# print(data.shape);
    
    

# return the batch_uid,batch_iid,batch_labels
def gen_batch(data,batch_size):
    data_size=data.shape[0];
    composed_batch=data[np.random.choice(data_size,batch_size),:];
    return composed_batch;
    
    
# to generated an ordered batch according to the data set seq    
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
