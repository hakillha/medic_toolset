import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

with tf.device("/device:cpu:0"):
    a = tf.constant([1])

with tf.device("/device:gpu:0"):
    b = a + 1

sess = tf.Session()
print(sess.run(b))
# print(sess.run(a))