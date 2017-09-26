import tensorflow as tf
import numpy as np

sess = tf.Session()
new_saver = tf.train.import_meta_graph('checkpoint/model_ckpt-1000.meta')
new_saver.restore(sess, 'checkpoint/model_ckpt-1000')

variables = tf.trainable_variables()

for ele in variables:
    print(ele)
weightfc1 = sess.run('W_fc1:0')
bias1 = sess.run('b_fc1:0')
weightfc2 = sess.run('W_fc2:0')
bias2 = sess.run('b_fc2:0')
np.savetxt('weights_fc1.csv', weightfc1, delimiter = ',')
np.savetxt('weights_fc2.csv', weightfc2, delimiter = ',')
np.savetxt('bias1.csv', bias1, delimiter = ',')
np.savetxt('bias2.csv', bias2, delimiter = ',')

