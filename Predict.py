import Model
import tensorflow as tf
import FaceInput
import numpy as np

with tf.Session() as sess:
    output = Model.FaceNet()
    img = FaceInput.GetOneImage('test/015-02.tif')
    saver = tf.train.Saver()
    saver.restore(sess,  tf.train.latest_checkpoint('checkpoint/'))
    result=sess.run(output,feed_dict={Model.x: img, Model.keep_prob: 1.0})
    result=np.argmax(result)
    print('The image is extracted from Person %d'%(result+1))