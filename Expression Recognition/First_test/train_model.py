import datetime
import time
import numpy as np

import tensorflow as tf

import pandas as pd

file_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')

#set the input tensor and the label tensor
X = tf.placeholder(tf.float32, [None, 48 * 48])
Y = tf.placeholder(tf.float32, [None, 7])

#control the dropout rate after the pool layer
keep_prob = tf.placeholder(tf.float32)


#set the train array
train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

# read data set
data = pd.read_csv('./dataset/fer2013.csv', dtype='a')
label = np.array(data['emotion'])
image_data = np.array(data['pixels'])
data_type = np.array(data['Usage'])

N_sample = label.size
Face_data = np.zeros((N_sample, 48 * 48))
Face_label = np.zeros((N_sample, 7), dtype=int)
for i in range(N_sample):
    x = image_data[i]
    x = np.fromstring(x, dtype=float, sep=' ')
#   Normalize the image
    x /= 256
    Face_data[i] = x
    Face_label[i, int(label[i])] = 1

#divide the dateset ,part of them is used to train and part of them is used to validate
train_data_x = Face_data[0:34000, :]
train_data_y = Face_label[0:34000, :]
test_data_x = Face_data[34001:, :]
test_data_y = Face_label[34001:, :]

#set the frequency of updating the weight and b
batch_size = 60

num_batch = len(train_data_x) // batch_size


test_batch_size = (len(test_data_x) // 100)



def expression_cnn():
    x = tf.reshape(X, shape=[-1, 48, 48, 1], )
    # 6 conv layers and a fully connect layer
    #set the conventional kernel and the Number of neurons（64）
    w_c1 = tf.Variable(tf.random_normal([7, 7, 1, 64], stddev=0.01))
    b_c1 = tf.Variable(tf.zeros([64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.01))
    b_c2 = tf.Variable(tf.zeros([128]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    # conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
    b_c3 = tf.Variable(tf.zeros([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv3 = tf.nn.dropout(conv3, keep_prob)

    w_c4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
    b_c4 = tf.Variable(tf.zeros([256]))
    conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    # conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c5 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
    b_c5 = tf.Variable(tf.zeros([256]))
    conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5))
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c6 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
    b_c6 = tf.Variable(tf.zeros([512]))
    conv6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5, w_c6, strides=[1, 1, 1, 1], padding='SAME'), b_c6))
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv6 = tf.nn.dropout(conv6, keep_prob)

    # fully connect layer
    w_d = tf.Variable(tf.random_normal([3 * 3 * 512, 1024], stddev=0.01))
    b_d = tf.Variable(tf.zeros([1024]))
    dense = tf.reshape(conv6, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    #
    # w_d = tf.Variable(tf.random_normal([4096, 1024], stddev=0.01))
    # b_d = tf.Variable(tf.zeros([1024]))
    # dense = tf.reshape(dense, [-1, w_d.get_shape().as_list()[0]])
    # dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    # dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(tf.random_normal([1024, 7], stddev=0.01))
    b_out = tf.Variable(tf.zeros([7]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def train():
    out = expression_cnn()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=Y))
    #set the method of learning rate and the Update strategy
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
    # global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    # rate = tf.train.exponential_decay(5e-5, global_step, decay_steps=500, decay_rate=0.97, staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    #caculate the accuracy of the output
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1)), tf.float32))

    # TensorBoard set the tensorboard lable which is used to show the changes in chart
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        pathstring = './log/' + file_time
        summary_writer = tf.summary.FileWriter(pathstring, graph=tf.get_default_graph())

        for e in range(200):
            for i in range(num_batch):
                batch_x = train_data_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_data_y[i * batch_size: (i + 1) * batch_size]
#         caculate the accauacy and the loss
                _, loss_, acc_, = sess.run([optimizer, loss, accuracy],
                                           feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
                # set the log for every steps
                # summary_writer.add_summary(summary, e * num_batch + i)
                print(e * num_batch + i, loss_, acc_, num_batch, e, )
                if (e * num_batch + i) % 100 == 0:

                    acc, summary = sess.run([accuracy, merged_summary_op],
                                            feed_dict={X: test_data_x, Y: test_data_y, keep_prob: 1.})
                    summary_writer.add_summary(summary, e * num_batch + i)
                    print(e * num_batch + i, acc)
        filepath = './model/' + file_time + '/convolutional.ckpt'
        saver.save(sess, filepath)

train()
