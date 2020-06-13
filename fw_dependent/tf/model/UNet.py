import tensorflow as tf

def contract_block(in_feat, filters, downsample, training):
    if downsample:
        in_feat = tf.layers.max_pooling2d(in_feat, (2, 2), 2)
    in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same", activation=tf.nn.relu)
    in_feat = tf.layers.batch_normalization(in_feat, training=training)
    in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same", activation=tf.nn.relu)
    return tf.layers.batch_normalization(in_feat, training=training)

def expand_block(in_feat, skip_feat, filters, upsample, training):
    if upsample:
        in_feat = tf.layers.conv2d_transpose(in_feat, filters, (2, 2), 2)
        in_feat = tf.layers.batch_normalization(in_feat, training=training)
    in_feat = tf.layers.conv2d(tf.concat(values=[in_feat, skip_feat], 
        axis=-1), filters, (3, 3), strides=1, padding="same", activation=tf.nn.relu)
    in_feat = tf.layers.batch_normalization(in_feat, training=training)
    in_feat = tf.layers.conv2d(in_feat, filters, (3, 3), strides=1, padding="same", activation=tf.nn.relu)
    return tf.layers.batch_normalization(in_feat, training=training)

def UNet(in_im, cfg, training=True):
    with tf.variable_scope("unet", 
        # initializer=tf.constant_initializer(0), 
        initializer=tf.random_normal_initializer(.0, .02),
        regularizer=tf.keras.regularizers.l2(cfg.network["weight_decay"])
        ):
        down1 = contract_block(in_im, 64, False, training)
        down2 = contract_block(down1, 128, True, training)
        down3 = contract_block(down2, 256, True, training)
        x = contract_block(down3, 512, True, training)
        x = expand_block(x, down3, 256, True, training)
        x = expand_block(x, down2, 128, True, training)
        x = expand_block(x, down1, 64, True, training)
        x = tf.layers.conv2d(x, 2, (1, 1), 1, activation=tf.nn.relu)
        # Use a dict to return res in case we have multiple outputs
        # return {"seg_map": tf.layers.conv2d(x, 1, (1, 1), 1, activation=tf.nn.sigmoid)}
        return {"seg_map": tf.layers.conv2d(x, 1, (1, 1), 1)}
