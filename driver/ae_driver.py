# an program for running ae and testing its performance
#  
import numpy as np
import tensorflow as tf

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import ae.denoising as autoencoder
import feeder.toy_random as feeder
# driver for this autoencoder only uses random embedding matrix
# which is also of some kind of importance for future research

flags=tf.app.flags;
FLAGS=flags.FLAGS;

flags.DEFINE_integer('in_dim',10,'the representation no dimension');

flags.DEFINE_integer('sample_num',1000,'the number of representations');

flags.DEFINE_integer('encode_dim',5,'the dimension of the encoder layer');

flags.DEFINE_integer('max_iter',5000000,'the max iteration numbers');

flags.DEFINE_integer('rating',0.05,'the learning rating');

flags.DEFINE_integer('log_step',500,'steps for logging reconstruction error');

flags.DEFINE_float('epsilon',0.0000001,'bound for precision');

flags.DEFINE_integer('batch_size',100,'batch size');
# no evaluating part, since this is not evaluable
# no need to use mini-batch
feeder.initialize(FLAGS.sample_num,FLAGS.in_dim);


packed_data=tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,FLAGS.in_dim],name='input_layer');
# 必要なことしか設定じゃ無い
ae_net=autoencoder.DenoisingAE(packed_data,FLAGS.batch_size,[FLAGS.in_dim,FLAGS.encode_dim]);


# define the initiation op
init_op=tf.initialize_all_variables();





optimizer=tf.train.GradientDescentOptimizer(FLAGS.rating);

loss=ae_net.loss();

train_op=optimizer.minimize(loss);

sess=tf.Session();

sess.run(init_op);




average_loss=0.0;



old_loss=50000000.0;
average_loss=0.0;
min_loss=-1.0;
for i in range(FLAGS.max_iter):
    feed_dict={
        packed_data:feeder.gen_batch(FLAGS.batch_size)
    }
    _,loss_val=sess.run([train_op,loss],feed_dict=feed_dict);
    average_loss+=loss_val;
    if(i%FLAGS.log_step==0):
        average_loss=average_loss/FLAGS.log_step;
        print("Step %d Average Loss %.8f" %(i,average_loss));
        if(np.abs(old_loss-average_loss)<=FLAGS.epsilon):    
            print("Final Loss %.8f Min Loss %.8f" %(average_loss,min_loss));
            break;
        old_loss=average_loss;
        min_loss=min(average_loss,min_loss);
        average_loss=0.0; # reset to zero
    









#
