# which should implement the training procedure of
# an program for running ae and testing its performance

import numpy as np
import tensorflow as tf

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import dae.dae as deep_auto_encoder
import feeder.toy_random as feeder

# driver for this autoencoder only uses random embedding matrix
# which is also of some kind of importance for future research

flags=tf.app.flags;
FLAGS=flags.FLAGS;

flags.DEFINE_integer('in_dim',10,'the representation no dimension');

flags.DEFINE_integer('sample_num',100000,'the number of representations');

flags.DEFINE_integer('max_iter',5000000,'the max iteration numbers');

flags.DEFINE_integer('rating',0.05,'the learning rating');

flags.DEFINE_integer('log_step',500,'steps for logging reconstruction error');

flags.DEFINE_float('tolerance',0.0001,'argument for pretraining');

flags.DEFINE_float('epsilon',0.0000001,'bound for precision');

flags.DEFINE_integer('batch_size',512,'batch size');
# no evaluating part, since this is not evaluable
# no need to use mini-batch


feeder.initialize(FLAGS.sample_num,FLAGS.in_dim);

dae_shape=[FLAGS.in_dim,30,60,30,10];

packed_data=tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,FLAGS.in_dim],name='input_layer');
# 必要なことしか設定じゃ無い
dae_net=deep_auto_encoder.DeepAutoEncoder(packed_data,FLAGS.batch_size,dae_shape);


# define the initiation op
init_op=tf.initialize_all_variables();


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
