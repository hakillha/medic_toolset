import tensorflow as tf
from tensorflow.keras import layers

class InputLayer(layers.Layer):
    def __init__(self, cfg):
        super(InputLayer, self).__init__()
        H, W, C = cfg.im_size[1], cfg.im_size[0], cfg.im_size[2]
        self.C = C
        self.avg_pool = layers.AveragePooling2D(pool_size=(H, W), strides=1)
        self.dense1 = layers.Dense(cfg.network["squeeze_ratio"])

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = tf.reshape(x, [-1, self.C])
        return x
    


# def SEResUNet():
