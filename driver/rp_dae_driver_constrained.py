# a rating prediction oriented deep autoencoder driver
# adding individual probability to control the inner product process

#
import  numpy as np
import tensorflow as tf


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


# which should implement the training procedure of
# an program for running ae and testing its performance

# an deep autoencoder is certainly one of the most powerful atomic models for user in my opinion

import dae.dae as deep_auto_encoder
#　use the facebook graph to do the link predicion problem
import feeder.ml_toy as feeder

# specify the dataset version
feeder.initialize(version='100k');



flags=tf.app.flags;
FLAGS=flags.FLAGS;

flags.DEFINE_integer('in_dim',10,'the representation no dimension');

flags.DEFINE_integer('max_iter',5000000,'the max iteration numbers');

flags.DEFINE_integer('rating',1.0,'the learning rating');

flags.DEFINE_integer('log_step',500,'steps for logging reconstruction error');

flags.DEFINE_integer('u_size',feeder.u_size,'u instance number');

flags.DEFINE_integer('v_size',feeder.v_size,'v instance number');

flags.DEFINE_integer('mean',3,'for eval');
flags.DEFINE_integer('delta',2,'for eval');



#　only define it ahead, it will not be used until next week or next day maybe
flags.DEFINE_integer('m_u_size',1,'number of machine generating user instances');
flags.DEFINE_integer('m_v_size',1,'number of machine generating item instances');

flags.DEFINE_float('tolerance',0.0001,'argument for pretraining');

flags.DEFINE_float('epsilon',0.00001,'bound for precision');

flags.DEFINE_integer('batch_size',512,'batch size');


# no evaluating part, since this is not evaluable
# no need to use mini-batch




dae_shape=[FLAGS.in_dim,40,10];

u=tf.placeholder(shape=[FLAGS.batch_size],dtype=tf.int32,name='batch_u_id');
v=tf.placeholder(shape=[FLAGS.batch_size],dtype=tf.int32,name='batch_v_id');
y_=tf.placeholder(shape=[FLAGS.batch_size],dtype=tf.float32,name='batch_label');


_low=-6.0/(FLAGS.in_dim);
_high=6.0/(FLAGS.in_dim);
#　define the instance random embedding matrix
u_vec_matrix=tf.Variable(np.random.uniform(low=_low,high=_high,size=(FLAGS.u_size,FLAGS.in_dim)),dtype=tf.float32,name='u_vec_matrix');
v_vec_matrix=tf.Variable(np.random.uniform(low=_low,high=_high,size=(FLAGS.v_size,FLAGS.in_dim)),dtype=tf.float32,name='v_vec_matrix');

#
u_prob_matrix=tf.Variable(np.random.uniform(low=_low,high=_high,size=(FLAGS.u_size,FLAGS.in_dim-1)),dtype=tf.float32,name='u_prob_matrix');
v_prob_matrix=tf.Variable(np.random.uniform(low=_low,high=_high,size=(FLAGS.v_size,FLAGS.in_dim-1)),dtype=tf.float32,name='v_prob_matrix');
# look up embeds according to the
u_embed=tf.nn.embedding_lookup(u_vec_matrix,u,name='batch_u_embedding');
v_embed=tf.nn.embedding_lookup(v_vec_matrix,v,name='batch_v_embedding');


# the following nets should be pretrained respectively
# 必要なことしか設定じゃ無い

# the implementation of the code assumes that the two net have the same structure
dae_net_u=deep_auto_encoder.DeepAutoEncoder(u_embed,FLAGS.batch_size,dae_shape);
# define dae net b as a shadow of net a
dae_net_v=deep_auto_encoder.DeepAutoEncoder(v_embed,FLAGS.batch_size,dae_shape);



# shadows eval are used for evaluating
dae_net_u_eval=deep_auto_encoder.DeepAutoEncoder(u_vec_matrix,FLAGS.u_size,dae_shape,variables=dae_net_u.variables());
dae_net_v_eval=deep_auto_encoder.DeepAutoEncoder(v_vec_matrix,FLAGS.v_size,dae_shape,variables=dae_net_v.variables());



# get two matrix of embeddings from the dae_nets
u_embed_list=dae_net_u.out(); # each level of (batch_size,d_k);
v_embed_list=dae_net_v.out();
def cosine_sim(a_embed,b_embed):
    a_normalized=tf.nn.l2_normalize(a_embed,dim=1);
    b_normalized=tf.nn.l2_normalize(b_embed,dim=1); # normalize along the each vector
    return tf.diag_part(tf.matmul(a_normalized,b_normalized,transpose_b=True));

def _eval(cos_sim,mean,delta):
    return cos_sim*delta+mean;

def dot_eval(a_embed,b_embed):
    return tf.diag_part(tf.matmul(a_embed,b_embed,transpose_b=True));
# currently just use all level embedding to predict, in future, it should become a stochastic process
def prediction(a_embed_list,b_embed_list):
    pred=tf.zeros([FLAGS.batch_size],dtype=np.float32);
    for i in range(dae_net_u.hidden_layer_size):
        pred+=_eval(cosine_sim(a_embed_list[i],b_embed_list[i]),FLAGS.mean,FLAGS.delta);
    return pred/dae_net_u.hidden_layer_size;



pred=prediction(u_embed_list,v_embed_list); # pred should of [1]




sqr_loss=tf.reduce_mean(tf.pow(pred-y_,2));
#ce_loss=tf.reduce_mean(-y_*tf.log(pred));

# to return the RMSE
def evaluate():
    # to maintain a max-L list
    u_embed_list=dae_net_u_eval.out();
    v_embed_list=dae_net_v_eval.out();
    pred_mat=tf.zeros([FLAGS.u_size,FLAGS.v_size],dtype=np.float32);
    for i in range(dae_net_u.hidden_layer_size):
        u_normalized=tf.nn.l2_normalize(u_embed_list[i],dim=1);
        v_normalized=tf.nn.l2_normalize(v_embed_list[i],dim=1); # normalize along the each vector
        pred_mat+=_eval(tf.matmul(u_normalized,v_normalized,transpose_b=True),FLAGS.mean,FLAGS.delta);
    return pred_mat/dae_net_u.hidden_layer_size;


evaluate_op=evaluate();



optimizer=tf.train.GradientDescentOptimizer(FLAGS.rating);
# be careful the dae_net's variables are of (3*n), n is the hidden layer
u_var_list=dae_net_u.variables();

u_loss_list=dae_net_u.loss();

u_train_list=[];


for i in range(dae_net_u.hidden_layer_size):
    u_train_list.append(optimizer.minimize(u_loss_list[i],var_list=u_var_list[3*i:3*(i+1)]));



v_var_list=dae_net_v.variables();

v_loss_list=dae_net_v.loss();

v_train_list=[];


for i in range(dae_net_v.hidden_layer_size):
    v_train_list.append(optimizer.minimize(v_loss_list[i],var_list=v_var_list[3*i:3*(i+1)]));

# should never train the embedding matrix
train_supervised=optimizer.minimize(sqr_loss,var_list=u_var_list.extend(v_var_list));



sess=tf.Session();

# define the initiation op
init_op=tf.initialize_all_variables();

sess.run(init_op);



# pretraining procedure, level by level
for j in range(dae_net_u.hidden_layer_size):
    print("BEGIN LEVEL %d MAP %d -> %d" %(j+1,dae_shape[j],dae_shape[j+1]));
    old_loss=50000000.0;
    average_loss=0.0;
    for i in range(FLAGS.max_iter):
        feed_dict={
            u:np.random.choice(FLAGS.u_size,FLAGS.batch_size)
        };
        _,loss_val=sess.run([u_train_list[j],u_loss_list[j]],feed_dict=feed_dict);
        average_loss+=loss_val;
        if(i%FLAGS.log_step==0):
            average_loss=average_loss/FLAGS.log_step;
            print("LEVEL %d Step %d Average Loss %.8f" %(j+1,i,average_loss));
            if(np.abs(old_loss-average_loss)<=FLAGS.tolerance):
                print("Final Loss %.8f" %(average_loss));
                print("END LEVEL %d" %(j+1));
                break; # go training the next level
            old_loss=average_loss;
            average_loss=0.0; # reset to zero


# after finishing

# pretraining procedure, level by level
for j in range(dae_net_v.hidden_layer_size):
    print("BEGIN LEVEL %d MAP %d -> %d" %(j+1,dae_shape[j],dae_shape[j+1]));
    old_loss=50000000.0;
    average_loss=0.0;
    for i in range(FLAGS.max_iter):
        feed_dict={
            v:np.random.choice(FLAGS.v_size,FLAGS.batch_size)
        };
        _,loss_val=sess.run([v_train_list[j],v_loss_list[j]],feed_dict=feed_dict);
        average_loss+=loss_val;
        if(i%FLAGS.log_step==0):
            average_loss=average_loss/FLAGS.log_step;
            print("LEVEL %d Step %d Average Loss %.8f" %(j+1,i,average_loss));
            if(np.abs(old_loss-average_loss)<=FLAGS.tolerance):
                print("Final Loss %.8f" %(average_loss));
                print("END LEVEL %d" %(j+1));
                break; # go training the next level
            old_loss=average_loss;
            average_loss=0.0; # reset to zero
# supervised training procedure



old_loss=50000000.0;
average_loss=0.0;
for i in range(FLAGS.max_iter):
    packed_data=feeder.gen_batch(FLAGS.batch_size);
    feed_dict={
        u:packed_data[:,0],
        v:packed_data[:,1],
        y_:packed_data[:,2]
    };
    _,loss_val=sess.run([train_supervised,sqr_loss],feed_dict=feed_dict);
    average_loss+=loss_val;
    if(i%FLAGS.log_step==0):
        average_loss=average_loss/FLAGS.log_step;
        print("Step %d Average Loss %.8f" %(i,average_loss));
        pred_mat=sess.run([evaluate_op],feed_dict={});
        print(pred_mat);
        rmse=feeder.rmse(pred_mat[0]);
        print("RMSE: %f" % (rmse));
        if(np.abs(old_loss-average_loss)<=FLAGS.tolerance):
            print("Final Loss %.8f" %(average_loss));
            break; # go training the next level
        old_loss=average_loss;
        average_loss=0.0; # reset to zero




pred_mat=sess.run([evaluate_op],feed_dict={});
print(pred_mat);
rmse=feeder.rmse(pred_mat[0]);
print("RMSE: %f" % (rmse));


#
