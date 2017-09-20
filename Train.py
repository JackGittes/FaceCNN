import Model
import tensorflow as tf
import cnnconfig as cf
import FaceInput

batch_size = cf.batch_size
imgs,labels,_ = FaceInput.ReadFaceImg.GetDataset()

def Train():
    out = Model.FaceNet()
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Model.y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Model.y_ * tf.log(out), reduction_indices=[1]))
    total_l2_loss = tf.add_n(tf.get_collection('losses'), name='l2_losses')
    total_loss = tf.add(cross_entropy, total_l2_loss)

    train_step = tf.train.AdamOptimizer(1e-5).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(max_steps):
            batch_imgs, batch_labels = Model.GetNextBatch(batch_size, imgs, labels)
            if epoch % 100 == 0:
               train_accuracy = accuracy.eval(feed_dict={Model.x: batch_imgs, Model.y_: batch_labels, Model.keep_prob: 1.0})
               current_loss = total_loss.eval(feed_dict={Model.x: batch_imgs, Model.y_: batch_labels, Model.keep_prob: 1.0})
       #        current_loss =1
               print("epoch %d, training accuracy=%g, loss= %g" % (epoch, train_accuracy, current_loss))
            if epoch % 1000 == 0:
                saver = tf.train.Saver()
                saver.save(sess, 'checkpoint/model_ckpt', global_step= epoch)
            train_step.run(feed_dict={Model.x: batch_imgs, Model.y_: batch_labels, Model.keep_prob: 0.6})
Train()