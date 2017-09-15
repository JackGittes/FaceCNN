import tensorflow as tf
import scipy
import numpy as np
import os
import cnnconfig as cf

height = cf.height
width = cf.width
class_num = cf.class_num

def CNNLayers():
	##Input layer
	x = tf.placeholder(tf.float32,[None,height,width,3])
	y_ = tf.placeholder(tf.float32,[None,class_num])
	
	##Convolution and Pooling operation
	def conv2d(x,kernel):
		return tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding='SAME')
	def max_pool_2(x):
		return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		
	##Convolution layer 1
	kernel_1 = weight_variable([3,3,3,5])
	bias_1 = bias_variable([5])
	relu1 = tf.nn.relu(conv2d(x,kernel_1)+bias_1)
	poo11 = max_pool_2(relu1)

	##Convolution layer 2	
	kernel_2 = weight_variable([3,3,5,7])
	bias_2 = bias_variable([7])
	relu2 = tf.nn.relu(conv2d(pool1,kernel_2)+bias_2)
	poo12 = max_pool_2(relu2)
		
	##Convolution layer 3		
	kernel_3 = weight_variable([3,3,5,7])
	bias_3 = bias_variable([7])
	relu3 = tf.nn.relu(conv2d(pool2,kernel_3)+bias_3)
	poo13 = max_pool_2(relu3)
		
	##Full-connected layer 1
	W_fc1 = weight_variable([,1024])
	b_fc1 = bias_variable([1024])
	pool3_flat = tf.reshape(pool3,[-1,])
	h_fc1 = tf.nn.relu(tf.matmal(pool3_flat,W_fc1) + b_fc1)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
	
	##Full-connected layer 2
	W_fc2 = weight_variable([1024,class_num])
	b_fc2 = bias_variable([class_num])
	y_conv = softmax(tf.mutmal(h_fc1_drop,W_fc2)+b_fc2)
		
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		
#def GiveFaceName():
def GetNextBatch(batch_size , iter_num , imgs , labels):
	batch_start = iter_num*batch_size
	batch_end = batch_start + batch
	batch_imgs = imgs[batch_start : batch_end]
	batch_labels = labels[batch_start : batch_end]
	return batch_imgs , batch_labels
	
tf.global_variables_initializer().run()
	for i in range(20000):
		if i%100 == 0
			train_accuracy = accuracy.eval(feed_dict={x:batch_imgs,y_:batch_labels,keep_prob:1.0})
			print("step %d, training accuracy = %g"%(i,training_accuracy))
		train_step.run(feed_dict={x:batch_imgs,y_:batch_labels},keep_prob:0.5)
		
		
		
	