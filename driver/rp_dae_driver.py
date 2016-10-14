# a rating prediction oriented deep autoencoder driver 
# which use the training data of the facebook social graph to try to predict the possible linkage in the
# test set. Use 'Precision' to evaluate it
import  numpy as np
import tensorflow as tf


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


# which should implement the training procedure of
# an program for running ae and testing its performance

import numpy as np
import tensorflow as tf

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import dae.dae as deep_auto_encoder

#　use the facebook graph to do the link predicion problem
import feeder.facebook_toy as feeder




flags=tf.app.flags;
FLAGS=flags.FLAGS;

flags.DEFINE_integer('in_dim',10,'the representation no dimension');

flags.DEFINE_integer('max_iter',5000000,'the max iteration numbers');

flags.DEFINE_integer('rating',0.05,'the learning rating');

flags.DEFINE_integer('log_step',500,'steps for logging reconstruction error');

flags.DEFINE_integer('u_size',feeder.u_size,'instance number');

#　only define it ahead, it will not be used until next week or next day maybe
flags.DEFINE_integer('m_size',1,'number of machine generating the instances');


flags.DEFINE_float('tolerance',0.0001,'argument for pretraining');

flags.DEFINE_float('epsilon',0.00001,'bound for precision');

flags.DEFINE_integer('batch_size',512,'batch size');

flags.DEFINE_integer('L',5000,'the base number for evaluating');
# no evaluating part, since this is not evaluable
# no need to use mini-batch




feeder.initialize();

dae_shape=[FLAGS.in_dim,40,10];

u_a=tf.placeholder(shape=[FLAGS.batch_size],dtype=tf.int32,name='batch_u_a');
u_b=tf.placeholder(shape=[FLAGS.batch_size],dtype=tf.int32,name='batch_u_b');
y_=tf.placeholder(shape=[FLAGS.batch_size],dtype=tf.float32,name='batch_label');


_low=-6.0/(FLAGS.in_dim);
_high=6.0/(FLAGS.in_dim);
#　define the instance random embedding matrix
u_vec_matrix=tf.Variable(np.random.uniform(low=_low,high=_high,size=(FLAGS.u_size,FLAGS.in_dim)),dtype=tf.float32,name='u_vec_matrix');
# look up embeds according to the
u_a_embed=tf.nn.embedding_lookup(u_vec_matrix,u_a,name='batch_u_a_embedding');
u_b_embed=tf.nn.embedding_lookup(u_vec_matrix,u_b,name='batch_u_b_embedding');



# 必要なことしか設定じゃ無い
dae_net_a=deep_auto_encoder.DeepAutoEncoder(u_a_embed,FLAGS.batch_size,dae_shape);
# define dae net b as a shadow of net a
dae_net_b=deep_auto_encoder.DeepAutoEncoder(u_b_embed,FLAGS.batch_size,dae_shape,variables=dae_net_a.variables());

# shadows eval are used for evaluating
dae_net_eval=deep_auto_encoder.DeepAutoEncoder(u_vec_matrix,FLAGS.u_size,dae_shape,variables=dae_net_a.variables());
# get two matrix of embeddings from the dae_nets
u_a_embed_list=dae_net_a.out(); # each level of (batch_size,d_k);
u_b_embed_list=dae_net_b.out();
def cosine_sim(a_embed,b_embed):
    a_normalized=tf.nn.l2_normalize(a_embed,dim=1);
    b_normalized=tf.nn.l2_normalize(b_embed,dim=1); # normalize along the each vector
    return tf.diag_part(tf.matmul(a_normalized,b_normalized,transpose_b=True));
# currently just use all level embedding to predict, in future, it should become a stochastic process
def prediction(a_embed_list,b_embed_list):
    pred=tf.zeros([FLAGS.batch_size],dtype=np.float32);
    for i in range(dae_net_a.hidden_layer_size):
        pred+=cosine_sim(a_embed_list[i],b_embed_list[i]);
    return pred/dae_net_a.hidden_layer_size;



pred=prediction(u_a_embed_list,u_b_embed_list); # pred should of [1]




sqr_loss=tf.reduce_mean(tf.pow(pred-y_,2));
ce_loss=tf.reduce_mean(-y_*tf.log(pred));

# to return the L-precision
def evaluate():
    # to maintain a max-L list
    a_embed_list=dae_net_eval.out();
    pred_mat=tf.zeros([FLAGS.u_size,FLAGS.u_size],dtype=np.float32);
    for i in range(dae_net_a.hidden_layer_size):
        a_normalized=tf.nn.l2_normalize(a_embed_list[i],dim=1);
        pred_mat+=tf.matmul(a_normalized,a_normalized,transpose_b=True);
    return pred_mat/dae_net_a.hidden_layer_size;


evaluate_op=evaluate();




# be careful the dae_net's variables are of (3*n), n is the hidden layer
total_var_list=dae_net_a.variables();



optimizer=tf.train.GradientDescentOptimizer(FLAGS.rating);

loss_list=dae_net_a.loss();

train_list=[];


for i in range(dae_net_a.hidden_layer_size):
    train_list.append(optimizer.minimize(loss_list[i],var_list=total_var_list[3*i:3*(i+1)]));



# should never train the embedding matrix
train_supervised=optimizer.minimize(sqr_loss,var_list=total_var_list);

sess=tf.Session();

# define the initiation op
init_op=tf.initialize_all_variables();

sess.run(init_op);



# pretraining procedure, level by level
for j in range(dae_net_a.hidden_layer_size):
    print("BEGIN LEVEL %d MAP %d -> %d" %(j+1,dae_shape[j],dae_shape[j+1]));
    old_loss=50000000.0;
    average_loss=0.0;
    for i in range(FLAGS.max_iter):
        feed_dict={
            u_a:np.random.choice(FLAGS.u_size,FLAGS.batch_size)
        };
        _,loss_val=sess.run([train_list[j],loss_list[j]],feed_dict=feed_dict);
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

# supervised training procedure

old_loss=50000000.0;
average_loss=0.0;
for i in range(FLAGS.max_iter):
    packed_data=feeder.gen_batch(FLAGS.batch_size);
    feed_dict={
        u_a:packed_data[:,0],
        u_b:packed_data[:,1],
        y_:packed_data[:,2]
    };
    _,loss_val=sess.run([train_supervised,sqr_loss],feed_dict=feed_dict);
    average_loss+=loss_val;
    if(i%FLAGS.log_step==0):
        average_loss=average_loss/FLAGS.log_step;
        print("Step %d Average Loss %.8f" %(i,average_loss));
        pred_mat=sess.run([evaluate_op],feed_dict={});
        print(pred_mat);
        train_hit,test_hit=feeder.precision(pred_mat,FLAGS.L);
        print("Precision: %f,%f" % (train_hit,test_hit));
        if(np.abs(old_loss-average_loss)<=FLAGS.tolerance):
            print("Final Loss %.8f" %(average_loss));
            break; # go training the next level
        old_loss=average_loss;
        average_loss=0.0; # reset to zero




pred_mat=sess.run([evaluate_op],feed_dict={});
print(pred_mat);
train_hit,test_hit=feeder.precision(pred_mat,FLAGS.L);
print("Precision: %f,%f" % (train_hit,test_hit));


#
