import tensorflow as tf
import cnnconfig as cf

height = cf.height
width = cf.width
class_num = cf.class_num
batch_size = cf.batch_size

x=tf.placeholder(tf.float32,[None,width,height,3])
y_=tf.placeholder(tf.float32,[None,class_num])
keep_prob = tf.placeholder(tf.float32)

def variable_with_weight_loss(shape, stddev, wl, name):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev),name=name)
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def conv2d(x, kernel):
	return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

def FaceNet():
	##Convolution layer 1
	kernel_1 = variable_with_weight_loss([3,3,3,64], stddev=0.01, wl=0.0,name = 'kernel_1')
	bias_1 = tf.Variable(tf.constant(0.01,shape=[64]),name='bias_1')
	relu1 = tf.nn.relu(conv2d(x,kernel_1)+bias_1)
	pool1= tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	##Convolution layer 2	
	kernel_2 = variable_with_weight_loss([3,3,64,64], stddev=0.01, wl=0.0,name='kernel_2')
	bias_2 = tf.Variable(tf.constant(0.01,shape=[64]),name='bias_2')
	relu2 = tf.nn.relu(conv2d(pool1,kernel_2)+bias_2)

		
	##Convolution layer 3		
	kernel_3 = variable_with_weight_loss([3,3,64,64], stddev=0.01,wl= 0.0,name='kernel_3')
	bias_3 = tf.Variable(tf.constant(0.01,shape=[64]),name='bias_3')
	relu3 = tf.nn.relu(conv2d(relu2,kernel_3)+bias_3)
	pool3 = tf.nn.max_pool(relu3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	##Convolution layer 4
	kernel_4 = variable_with_weight_loss([1,1,64,64], stddev=0.01, wl=0.0,name='kernel_4')
	bias_4 = tf.Variable(tf.constant(0.01,shape=[64]),name='bias_4')
	relu4 = tf.nn.relu(conv2d(pool3,kernel_4)+bias_4)
	pool4 = tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	##Convolution layer 5
	kernel_5 = variable_with_weight_loss([1,1,64,32], stddev=0.01, wl=0.0,name='kernel_5')
	bias_5 = tf.Variable(tf.constant(0.01,shape=[32]),name='bias_5')
	relu5 = tf.nn.relu(conv2d(pool4,kernel_5)+bias_5)
#	pool5 = tf.nn.max_pool(relu5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	##Full-connected layer 1
	W_fc1 = variable_with_weight_loss([10*10*32,64], stddev=0.01, wl=0.004,name='W_fc1')
	b_fc1 = tf.Variable(tf.constant(0.01,shape=[64]),name='b_fc1')
	pool_flat = tf.reshape(relu5,[-1,10*10*32])
	h_fc1 = tf.nn.relu(tf.matmul(pool_flat,W_fc1) + b_fc1)

	h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
	
	##Full-connected layer 2
	W_fc2 = variable_with_weight_loss([64,class_num], stddev=0.01, wl=0.004,name='W_fc2')
	b_fc2 = tf.Variable(tf.constant(0.01,shape=[class_num]),name='b_fc2')
	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
	return y_conv