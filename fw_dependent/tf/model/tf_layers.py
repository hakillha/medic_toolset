import tensorflow as tf

from .ASEUNet import SEResUNet

def hbloss_dice_focal_v0(pred, input_ann):
    prob = tf.math.sigmoid(pred)
    tp = tf.math.reduce_sum(tf.math.multiply(prob, input_ann))
    fn = tf.math.reduce_sum(tf.math.multiply(tf.constant([1.0]) - prob, input_ann))
    fp = tf.math.reduce_sum(tf.math.multiply(prob, tf.constant([1.0]) - input_ann))
    loss_dice = tp / (tp + tf.constant(.5) * fn + tf.constant(.5) * fp + tf.constant([1e-5]))
    lambda_ = tf.constant([.5])
    loss_focal = lambda_ * tf.reduce_mean(tf.math.multiply(tf.math.log(prob),
        tf.math.multiply(input_ann, tf.math.square(tf.constant([1.0]) - prob))))
    return tf.constant([1.0]) - loss_dice - loss_focal

def generalized_dice_loss(pred, input_ann):
    prob = tf.math.sigmoid(pred)
    rp = tf.reduce_sum(tf.math.multiply(prob, input_ann))
    wl = tf.reduce_sum(input_ann)
    wl = 1.0 / (tf.math.square(wl) + 1e-5)
    r_p = tf.reduce_sum(prob + input_ann)
    return 1.0 - 2.0 * tf.reduce_sum(tf.math.multiply(wl, rp)) / tf.reduce_sum(tf.math.multiply(wl, r_p)) + 1e-5

def hbloss_dice_focal_v1(pred, input_ann):
    prob = tf.math.sigmoid(pred)
    alpha = .5
    beta = .5
    p0 = prob
    p1 = 1.0 - prob
    g0 = input_ann
    g1 = 1.0 - input_ann
    num = tf.reduce_sum(p0 * g0 * (1 - p0) * (1 - p0))
    den = num + alpha * tf.reduce_sum(p0 * g1) + beta * tf.reduce_sum(p1 * g0)
    T = tf.reduce_sum(num / den + 1e-6)
    return 1.6 * (1.0 - T)

def dice_loss(pred, input_ann):
    prob = tf.math.sigmoid(pred)
    EPS = 1e-5
    dice = 2.0 * (tf.reduce_sum(prob * input_ann) + EPS) / (tf.reduce_sum(prob * prob) + tf.reduce_sum(input_ann * input_ann) + EPS)
    return 1.0 - dice

def focal_loss(pred, input_ann):
    prob = tf.math.sigmoid(pred)
    prob_ = 1.0 - prob
    y_true_loss = tf.cast(tf.equal(input_ann, True), tf.float32) * prob_ * prob_ * tf.math.log(prob)
    y_false_loss = tf.cast(tf.equal(input_ann, False), tf.float32) * prob * prob * tf.math.log(prob_)
    return -tf.reduce_mean(y_true_loss + y_false_loss)

class tf_model():
    def __init__(self, args, cfg, num_batches=None):
        self.input_im = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], 1), name="input_im")
        if cfg.num_class == 1:
            self.pred = SEResUNet(self.input_im, num_classes=1, reduction=8, name_scope="SEResUNet")
        else:
            self.pred = SEResUNet(self.input_im, num_classes=cfg.num_class + 1, reduction=8, name_scope="SEResUNet")
        
        if args.mode == "train":
            if cfg.num_class == 1:
                # For some reason float32 is working
                self.input_ann = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], 1))
                if cfg.loss == "sigmoid":
                    self.ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_ann, logits=self.pred)
                    self.loss = tf.reduce_mean(self.ce)
                elif cfg.loss == "hbloss_dice_focal":
                    # self.loss = hbloss_dice_focal(self.pred, self.input_ann)
                    self.loss = hbloss_dice_focal_v1(self.pred, self.input_ann)
                elif cfg.loss == "generalized_dice_loss":
                    self.loss = generalized_dice_loss(self.pred, self.input_ann)
                elif cfg.loss == "dice_loss":
                    self.loss = dice_loss(self.pred, self.input_ann)
                elif cfg.loss == "focal":
                    self.loss = focal_loss(self.pred, self.input_ann)
                elif cfg.loss == "hbloss_dice_focal_v2":
                    self.loss = dice_loss(self.pred, self.input_ann) + .5 * focal_loss(self.pred, self.input_ann)
                elif cfg.loss == "hbloss_dice_ce":
                    ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_ann, logits=self.pred))
                    self.loss = dice_loss(self.pred, self.input_ann) + .1 * ce
            else:
                if cfg.loss == "softmax":
                    self.input_ann = tf.placeholder(tf.int32, shape=(None, cfg.im_size[0], cfg.im_size[1]))
                    self.ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_ann, logits=self.pred)
                    self.loss = tf.reduce_mean(self.ce)
                elif cfg.loss == "sigmoid":
                    self.input_ann = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], cfg.num_class + 1))
                    self.ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_ann, logits=self.pred)
                    self.loss = tf.reduce_mean(self.ce)
                    
            global_step = tf.Variable(0, trainable=False)
            assert num_batches, "Please provide batch size for training mode!"
            steps = [num_batches * epoch_step for epoch_step in cfg.optimizer["epoch_to_drop_lr"]]
            print(f"Steps to drop LR: {steps}")
            learning_rate = tf.train.piecewise_constant(global_step, steps, cfg.optimizer["lr"])
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

