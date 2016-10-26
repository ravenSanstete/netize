# a feeder for the movie lens 100k data set
# to implement the feeder for each
import zipfile
import os,sys,inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

PATH='ml-1m.zip';
train_path='ml-1m/train.dat';
test_path='ml-1m/test.dat'
zip_name='ml-1m.zip';


# add some configuration dict
# format str:list[zip_fname,train_fname,test_fname,u_size,i_size,delim]

# movie_lens dataset, id begins from 1
config={
    '1m': {
        'zip_name':'movie_lens/ml-1m.zip',
        'train_name': 'ml-1m/train.dat',
        'test_name': 'ml-1m/test.dat',
        'u_size': 6040,
        'v_size': 3952,
        'delim':'::'
     },
   '100k':{
        'zip_name':'movie_lens/ml-100k.zip',
        'train_name': 'ml-100k/ua.base',
        'test_name': 'ml-100k/ua.test',
        'u_size': 943,
        'v_size': 1682,
        'delim': '\t'
     }
}









train_set=[];
test_set=[];
u_size=0;
v_size=0;



# version could be '1m', '100k', '10m'
def initialize(version='1m'):
    global train_set;
    global test_set;
    global u_size;
    global v_size;
    u_size=config[version]['u_size'];
    v_size=config[version]['v_size'];
    train_set=build_dataset(read_data(config[version]['zip_name'],config[version]['train_name'],config[version]['delim']));
    test_set=build_dataset(read_data(config[version]['zip_name'],config[version]['test_name'],config[version]['delim']));
    pass;


# unzipped the file and then read in the u.data
def read_data(zip_path,filename,delimiter):
    print('unzip dataset');
    with zipfile.ZipFile(os.path.join(currentdir,zip_path)) as ml_zip:
        with ml_zip.open(filename) as ml:
            line=ml.read().decode('utf-8');
            raw=line.replace(delimiter,'\t').replace('\n','\t').split('\t');
    return raw;




# to construct it as a tensor data set
# n%4 : 0 uid, 1 iid, 2 rating, 3 timestamp
def build_dataset(raw,modu=4):
    print('build dataset');
    try:
        assert len(raw)%modu==0;
    except Exception as e:
        raw=raw[:-1]; # truncate the last empty entry introduced by splitting process

    ds_size=len(raw)//modu;
    data=np.zeros(shape=(ds_size,modu),dtype=np.int32); # define a custom tensor;
    data_count=0;
    row_count=0;
    for field in raw:
        ind=data_count%modu;
        if(ind<2):
            field=int(field)-1; # to convert the id into cs convention
        data[row_count,ind]=field;
        data_count+=1;
        if(data_count%modu==0):
            row_count+=1;
    return data

#
# data=build_dataset(raw);
#
# print(data.shape);



# return the batch_uid,batch_iid,batch_labels
def gen_batch(batch_size,train=True):
    if(train):
        ds=train_set;
    else:
        ds=test_set;
    batch=ds[np.random.choice(ds.shape[0],batch_size),:];
    return batch;


# to generated an ordered batch according to the data set seq
def gen_ordered_batch(data,beg,batch_size):
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





def gen_mat(u_len,i_len,test=True):
    if(test):
        data_set=test_set;
    else:
        data_set=train_set;
    mat=np.zeros((u_len,i_len),dtype=np.float32);
    ind=np.zeros((u_len,i_len),dtype=np.int32);
    for i in range(data_set.shape[0]):
        row=data_set[i];
        mat[row[0]-1][row[1]-1]=row[2];
        ind[row[0]-1][row[1]-1]=1;
    return mat,ind;

# compute the rmse according to the prediction matrix passed in
def rmse(pred_mat):
    test_mat,ind_mat=gen_mat(pred_mat.shape[0],pred_mat.shape[1]);
    return np.sqrt(np.sum(np.power(ind_mat*(pred_mat-ind_mat),2))/test_set.shape[0]);
    pass;

#　simple test for the correctness of this feeder
if  __name__=='__main__':
    initialize(version='100k');
    print(gen_batch(batch_size=100));


















#
