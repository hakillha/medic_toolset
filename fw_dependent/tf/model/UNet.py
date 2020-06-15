import tensorflow as tf

from .misc import get_norm

def contract_block(in_feat, name, filters, downsample, norm_layer, training):
    with tf.variable_scope(name):
        if downsample:
            in_feat = tf.layers.max_pooling2d(in_feat, (2, 2), 2)
        in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same", activation=tf.nn.relu)
        in_feat = get_norm(norm_layer, f"{norm_layer}_1", training)(in_feat)
        in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same", activation=tf.nn.relu)
        return get_norm(norm_layer, f"{norm_layer}_2", training)(in_feat)

def expand_block(in_feat, skip_feat, name, filters, upsample, norm_layer, training):
    with tf.variable_scope(name):
        if upsample:
            in_feat = tf.layers.conv2d_transpose(in_feat, filters, (2, 2), 2)
            in_feat = get_norm(norm_layer, f"{norm_layer}_1", training)(in_feat)
        in_feat = tf.layers.conv2d(tf.concat(values=[in_feat, skip_feat], 
            axis=-1), filters, (3, 3), strides=1, padding="same", activation=tf.nn.relu)
        in_feat = get_norm(norm_layer, f"{norm_layer}_2", training)(in_feat)
        in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same", activation=tf.nn.relu)
        return get_norm(norm_layer, f"{norm_layer}_3", training)(in_feat)

def UNet(in_im, cfg, training=True):
    with tf.variable_scope("unet", 
        # initializer=tf.constant_initializer(0), 
        initializer=tf.random_normal_initializer(.0, .02),
        regularizer=tf.keras.regularizers.l2(cfg.network["weight_decay"])
        ):
        down1 = contract_block(in_im, "down1", 64, False, cfg.network["norm_layer"], training)
        down2 = contract_block(down1, "down2", 128, True, cfg.network["norm_layer"], training)
        down3 = contract_block(down2, "down3", 256, True, cfg.network["norm_layer"], training)
        x = contract_block(down3, "down4", 512, True, cfg.network["norm_layer"], training)
        x = expand_block(x, down3, "up1", 256, True, cfg.network["norm_layer"], training)
        x = expand_block(x, down2, "up2", 128, True, cfg.network["norm_layer"], training)
        x = expand_block(x, down1, "up3", 64, True, cfg.network["norm_layer"], training)
        x = tf.layers.conv2d(x, 2, (1, 1), 1, activation=tf.nn.relu)
        # Use a dict to return res in case we have multiple outputs
        # return {"seg_map": tf.layers.conv2d(x, 1, (1, 1), 1, activation=tf.nn.sigmoid)}
        return {"seg_map": tf.layers.conv2d(x, 1, (1, 1), 1)}
