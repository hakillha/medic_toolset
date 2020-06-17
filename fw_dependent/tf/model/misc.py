from functools import partial
from pprint import pprint

import tensorflow as tf
from tensorpack.callbacks import Callback

def get_norm(norm_name, name, training):
    """
        args:
            training: This is auto taken care of in tp
    """
    if norm_name == "BN_layers":
        return partial(tf.layers.batch_normalization, training=training, name=name)
    elif norm_name == "BN_layers":
        return partial(GroupNomr_layers, training=training, name=name)
    elif norm_name == "BN":
        return partial(BatchNorm, name=name)
    elif norm_name == "GN":
        return partial(GroupNorm, name=name)
    elif norm_name == "None":
        return tf.identity

def GroupNomr_layers(x, name, training, group=16):
    N, H, W, C = x.get_shape().as_list()
    assert C > group, f"#channels: {C}."
    assert C % group == 0, f"#channels: {C}."
    x = tf.reshape(x, [-1, H, W, C // group, group])
    x = tf.layers.batch_normalization(x, axis=[1, 2, 3], training=training, name=name)
    x = tf.reshape(x, [-1, H, W, C])
    return x

def GroupNorm(x, name, group=16, esp=1e-5, liupan=True):
    with tf.variable_scope(name):
        # tranpose: [bs,h,w,c] to [bs,c,h,w] 
        if not liupan:
            og_shape = tf.shape(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        assert C > group, f"#channels: {C}."
        assert C % group == 0, f"#channels: {C}."
        group_size = C // group
        x = tf.reshape(x, [-1, group, group_size, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.0))
        if not liupan:
            # YuxinWu implementation in tensorpack FRCNN sample
            new_shape = [1, group, group_size, 1, 1]
            gamma = tf.reshape(gamma, new_shape)
            beta = tf.reshape(beta, new_shape)
            output = tf.nn.batch_normalization(x, mean, var, beta, gamma, esp, name="output")
            output = tf.reshape(output, og_shape, name="output")
        else:
            # PanLiu implementation
            x = (x - mean) / tf.sqrt(var + esp)
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])
            output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
            output = tf.transpose(output, [0, 2, 3, 1])
        return output

def BatchNorm(x, name, esp=1e-5):
    with tf.variable_scope(name):
        # x: [bs, h, w, c]
        N, H, W, C = x.get_shape().as_list()
        mean, var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.0))
        new_shape = [1, 1, 1, C]
        gamma = tf.reshape(gamma, new_shape)
        beta = tf.reshape(beta, new_shape)
        output = tf.nn.batch_normalization(x, mean, var, beta, gamma, esp, name="output")
        return output

class BN_layers_update(Callback):
    # def __init__(self, cfg):
    #     self.cfg = cfg
    
    def _setup_graph(self):
        # self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="BN_layers")
        self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("\ntf.layers.BN update operations:")
        pprint(self.update_op)

    def _before_run(self, _):
        return tf.train.SessionRunArgs(fetches=self.update_op)

if __name__ == "__main__":
    import numpy as np
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    test_v = tf.random_normal((2, 384, 384, 64))
    test_v = GroupNorm(test_v)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # res_list = sess.run(test_v)
    # for res in res_list:
    #     print(res.shape)
    print(sess.run(test_v).shape)