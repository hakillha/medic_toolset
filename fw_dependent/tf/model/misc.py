from functools import partial
import tensorflow as tf

def get_norm(norm_name, name, training):
    if norm_name == "BN":
        return partial(tf.layers.batch_normalization, training=training)
    elif norm_name == "GN":
        return partial(GroupNorm, name=name)

def GroupNorm(x, name, group=16, esp=1e-5):
    with tf.variable_scope(name):
        # tranpose: [bs,h,w,c] to [bs,c,h,w] 
        og_shape = tf.shape(x)
        x = tf.transpose(x,[0,3,1,2])
        N, C, H, W = x.get_shape().as_list()
        assert C > group, f"#channels: {C}."
        assert C % group == 0, f"#channels: {C}."
        group_size = C // group
        x = tf.reshape(x, [-1, group, group_size, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.0))

        # YuxinWu implementation in tensorpack FRCNN sample
        # new_shape = [1, group, group_size, 1, 1]
        # gamma = tf.reshape(gamma, new_shape)
        # beta = tf.reshape(beta, new_shape)
        # output = tf.nn.batch_normalization(x, mean, var, beta, gamma, esp, name="output")
        # output = tf.reshape(output, og_shape, name="output")

        # PanLiu implementation
        x = (x - mean) / tf.sqrt(var + esp)
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        output = tf.transpose(output, [0, 2, 3, 1])
        
        return output

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