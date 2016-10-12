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

flags.DEFINE_float('epsilon',0.0000001,'bound for precision');

flags.DEFINE_integer('batch_size',512,'batch size');
# no evaluating part, since this is not evaluable
# no need to use mini-batch




feeder.initialize();

dae_shape=[FLAGS.in_dim,30,60,30,10];

u_a=tf.placeholder(shape=[FLAGS.batch_size],dtype=tf.int32,name='batch_u_a');
u_b=tf.placeholder(shape=[FLAGS.batch_size],dtype=tf.int32,name='batch_u_b');



_low=-6.0/(FLAGS.in_dim);
_high=6.0/(FLAGS.out_dim);
#　define the instance random embedding matrix
u_vec_matrix=tf.Variable(np.random.uniform(low=_low,high=_high,size=(FLAGS.u_size,FLAGS.in_dim)),dtype=tf.float32,name='u_vec_matrix');
# look up embeds according to the



# define the initiation op
init_op=tf.initialize_all_variables();
# 必要なことしか設定じゃ無い

dae_net=deep_auto_encoder.DeepAutoEncoder(packed_data,FLAGS.batch_size,dae_shape);

# be careful the dae_net's variables are of (3*n), n is the hidden layer
total_var_list=dae_net.variables();



optimizer=tf.train.GradientDescentOptimizer(FLAGS.rating);

loss_list=dae_net.loss();

train_list=[];


for i in range(dae_net.hidden_layer_size):
    train_list.append(optimizer.minimize(loss_list[i],var_list=total_var_list[3*i:3*(i+1)]));


sess=tf.Session();

sess.run(init_op);



# pretraining procedure, level by level
for j in range(dae_net.hidden_layer_size):
    print("BEGIN LEVEL %d MAP %d -> %d" %(j+1,dae_shape[j],dae_shape[j+1]));
    old_loss=50000000.0;
    average_loss=0.0;
    for i in range(FLAGS.max_iter):
        feed_dict={
            packed_data:feeder.gen_batch(FLAGS.batch_size)
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










#
