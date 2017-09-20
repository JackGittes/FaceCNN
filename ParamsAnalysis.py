import tensorflow as tf
import numpy as np

sess = tf.Session()
new_saver = tf.train.import_meta_graph('checkpoint/model_ckpt-2000.meta')
new_saver.restore(sess, 'checkpoint/model_ckpt-2000')

variables = tf.trainable_variables()

for ele in variables:
    print(ele)
weightfc1 = sess.run('Variable_6:0')
weightfc2 = sess.run('Variable_8:0')
#np.savetxt('weights_fc1.csv', weightfc1, delimiter = ',')
#np.savetxt('weights_fc2.csv', weightfc2, delimiter = ',')
print(weightfc1)

