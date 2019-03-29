import tensorflow as tf
import numpy as np

from zdaiot import AlexNet

batch_size = 32

images = tf.placeholder(tf.float32, [batch_size, 256, 256, 3])

net = AlexNet({'data': images})

with tf.Session() as sess:
    # Load the data
    sess.run(tf.global_variables_initializer())
    net.load('zdaiot.npy', sess)