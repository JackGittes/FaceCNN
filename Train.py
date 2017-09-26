import Model
import tensorflow as tf
import cnnconfig as cf
import FaceInput

batch_size = cf.batch_size
max_steps = cf.max_steps
imgs,labels,_ = FaceInput.ReadFaceImg.GetDataset()
check_epoch = cf.check_epoch
eval_epoch = cf.eval_epoch

def Train():
    out = Model.FaceNet()
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Model.y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cliped_out = tf.clip_by_value(out,1e-10,1e100)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Model.y_ * tf.log(cliped_out), reduction_indices=[1]))
    total_l2_loss = tf.add_n(tf.get_collection('losses'), name='l2_losses')
    total_loss = tf.add(cross_entropy, total_l2_loss)

    train_step = tf.train.AdamOptimizer(0.5*1e-4).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(max_steps):
            batch_imgs, batch_labels = FaceInput.GetNextBatch(batch_size, imgs, labels)
            if epoch % eval_epoch == 0:
               train_accuracy = accuracy.eval(feed_dict={Model.x: batch_imgs, Model.y_: batch_labels, Model.keep_prob: 1.0})
               current_loss = total_loss.eval(feed_dict={Model.x: batch_imgs, Model.y_: batch_labels, Model.keep_prob: 1.0})
               print("epoch %d, training accuracy=%g, loss= %g" % (epoch, train_accuracy, current_loss))
            if epoch % check_epoch == 0 and epoch>0:
                saver = tf.train.Saver()
                saver.save(sess, 'checkpoint/model_ckpt', global_step= epoch)
            train_step.run(feed_dict={Model.x: batch_imgs, Model.y_: batch_labels, Model.keep_prob: 0.5})

if __name__ == '__main__':
    print('The training process will finish after %d steps.'%(max_steps))
    print('The checkpoint is stored every %d epochs ...'%(check_epoch))
    Train()