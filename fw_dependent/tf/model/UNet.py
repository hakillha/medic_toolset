import tensorflow as tf

def contract_block(in_feat, filters, downsample, training):
    if downsample:
        in_feat = tf.layers.max_pooling2d(in_feat, (2, 2), 2)
    in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same")
    in_feat = tf.layers.batch_normalization(in_feat, training=training)
    in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same")
    return tf.layers.batch_normalization(in_feat, training=training)

def expand_block(in_feat, skip_feat, filters, upsample, training):
    if upsample:
        in_feat = tf.layers.conv2d_transpose(in_feat, filters, (2, 2), 2)
        in_feat = tf.layers.batch_normalization(in_feat, training=training)
    in_feat = tf.layers.conv2d(tf.concat(values=[in_feat, skip_feat], axis=-1), filters, (3, 3), strides=1, padding="same")
    in_feat = tf.layers.batch_normalization(in_feat, training=training)
    in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same")
    return tf.layers.batch_normalization(in_feat, training=training)

def UNet(in_im, cfg, training=True):
    with tf.variable_scope("unet", 
        initializer=tf.constant_initializer(0), 
        regularizer=tf.keras.regularizers.l2(cfg.network["weight_decay"])):
        down1 = contract_block(in_im, 64, False, training)
        down2 = contract_block(down1, 128, True, training)
        down3 = contract_block(down2, 256, True, training)
        x = contract_block(down3, 512, True, training)
        x = expand_block(x, down3, 256, True, training)
        x = expand_block(x, down2, 128, True, training)
        x = expand_block(x, down1, 64, True, training)
        return {"seg_map": tf.layers.conv2d(x, 1, (1, 1), 1)}

class tf_model():
    def __init__(self, cfg, training=True):
        with tf.device("/cpu:0"):
            self.in_im = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], 1), name="input_im")
            self.in_gt = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], 1), name="input_gt")
        log_prob = UNet(self.in_im, cfg, training)
        self.prob = tf.math.sigmoid(log_prob) 

# Test
if __name__ == "__main__":
    import os, sys
    import numpy as np
    sys.path.insert(0, "../../..")
    from fw_neutral.utils.config import Config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = Config()
    cfg.load_from_json("/rdfs/fast/home/sunyingge/pt_ground/configs/base_0526.json")
    model = tf_model(cfg)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res_list = sess.run([model.prob],
        feed_dict={model.in_im: np.random.normal(size=(1, 384, 384, 1)), 
            model.in_gt: np.random.normal(size=(1, 384, 384, 1))})
    for res in res_list:
        print(res.shape)