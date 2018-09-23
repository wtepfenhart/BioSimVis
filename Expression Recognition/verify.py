import numpy as np
import tensorflow as tf
from PIL import Image

#It is mainly for predicting the image of face and output the probaility and the index of label
keep_prob = tf.placeholder(tf.float32)

images=tf.placeholder(dtype=tf.float32,shape=[None,48,48,1])


def expression_cnn():
    x=images

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

    w_out = tf.Variable(tf.random_normal([1024, 7], stddev=0.01))
    b_out = tf.Variable(tf.zeros([7]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out




def Verify(image):
    out = expression_cnn()
    probability = tf.nn.softmax(out)
    predict_val,predict_index_val= tf.nn.top_k(probability,k=1)
    temp_image=Image.open(image).convert('L')
    temp_image=temp_image.resize((48,48),Image.ANTIALIAS)
    temp_image=np.asanyarray(temp_image)/255.0
    temp_image=temp_image.reshape([-1,48,48,1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        ckpt=tf.train.latest_checkpoint('./model/')
        if ckpt:
            saver.restore(sess,ckpt)
        predict_val,predict_index_val=sess.run([predict_val,predict_index_val],feed_dict={images:temp_image,keep_prob:1})
    print(predict_val,predict_index_val)


Verify('test.png')
