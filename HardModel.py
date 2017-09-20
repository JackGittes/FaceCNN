import tensorflow as tf
import numpy as np

sess = tf.Session()
new_saver = tf.train.import_meta_graph('checkpoint/model_ckpt-2000.meta')
new_saver.restore(sess, 'checkpoint/model_ckpt-2000')

variables = tf.trainable_variables()


x=tf.placeholder(tf.float32,[None,width,height,3])
y_=tf.placeholder(tf.float32,[None,class_num])

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.02)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.02,shape=shape)
	return tf.Variable(initial)

##Input layer
##Convolution and Pooling operation
def conv2d(x, kernel):
    return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


##Convolution layer 1
kernel_1 = tf.Variable(weights.initialized_value())
bias_1 = bias_variable([64])
relu1 = tf.nn.relu(conv2d(x, kernel_1) + bias_1)
# pool1= max_pool_2(relu1)

##Convolution layer 2
kernel_2 = weight_variable([3, 3, 64, 64])
bias_2 = bias_variable([64])
relu2 = tf.nn.relu(conv2d(relu1, kernel_2) + bias_2)
pool2 = max_pool_2(relu2)

##Convolution layer 3
kernel_3 = weight_variable([3, 3, 64, 64])
bias_3 = bias_variable([64])
relu3 = tf.nn.relu(conv2d(pool2, kernel_3) + bias_3)
pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')