import tensorflow as tf

import sys
sys.path.insert(0, "../../..")
from fw_dependent.tf.model.ASEUNet import SEResUNet
from fw_dependent.tf.model.UNet import UNet, UNet_v1, UNet_v1_9, UNet_v2

def choose_model(name):
    MODEL_MAP = {
        "SEResUNet": SEResUNet,
        "UNet": UNet,
        "UNet_v1": UNet_v1,
        "UNet_v1.9": UNet_v1_9,
        "UNet_v2": UNet_v2
    }
    return MODEL_MAP[name]

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

def build_loss(pred, input_im, input_ann, cfg):
    seg_map = pred["seg_map"]
    if cfg.num_class == 1:
        # For some reason float32 is working
        if cfg.loss == "sigmoid":
            ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_ann, logits=seg_map)
            loss = tf.reduce_mean(ce)
        elif cfg.loss == "hbloss_dice_focal":
            # loss = hbloss_dice_focal(seg_map, input_ann)
            loss = hbloss_dice_focal_v1(seg_map, input_ann)
        elif cfg.loss == "generalized_dice_loss":
            loss = generalized_dice_loss(seg_map, input_ann)
        elif cfg.loss == "dice_loss":
            loss = dice_loss(seg_map, input_ann)
        elif cfg.loss == "focal":
            loss = focal_loss(seg_map, input_ann)
        elif cfg.loss == "hbloss_dice_focal_v2":
            loss = dice_loss(seg_map, input_ann) + .5 * focal_loss(seg_map, input_ann)
        elif cfg.loss == "hbloss_dice_ce":
            ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_ann, logits=seg_map))
            loss = dice_loss(seg_map, input_ann) + .5 * ce
        elif cfg.loss == "hbloss_gendice_ce":
            ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_ann, logits=seg_map))
            loss = generalized_dice_loss(seg_map, input_ann) + .1 * ce
    else:
        if cfg.loss == "softmax":
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_ann, logits=seg_map)
            loss = tf.reduce_mean(ce)
        elif cfg.loss == "sigmoid":
            ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_ann, logits=seg_map)
            loss = tf.reduce_mean(ce)
    if cfg.network["reconstruct"]:
        loss += .1 * tf.losses.mean_squared_error(input_im, pred["recon_map"])
    if cfg.network["weight_decay"]:
        loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g != None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        if len(grads):
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
        else:
            grad = None
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class tf_model():
    def __init__(self, cfg, training, gpus=None, num_batches=None):
        with tf.device('/cpu:0'):
            self.in_im = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], 1), name="input_im")
            if cfg.num_class == 1:
                self.in_gt = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], 1))
            else:
                if cfg.loss == "softmax":
                    self.in_gt = tf.placeholder(tf.int32, shape=(None, cfg.im_size[0], cfg.im_size[1]))
                elif cfg.loss == "sigmoid":
                    self.in_gt = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], cfg.num_class + 1))
            if training:   
                # if gpus:
                global_step = tf.get_variable("global_step", 
                    [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
                # else:
                #     global_step = tf.Variable(0, False) # For backward compatibility
                self.global_step = global_step
                assert num_batches, "Please provide batch size for training mode!"
                steps = [num_batches * epoch_step for epoch_step in cfg.optimizer["epoch_to_drop_lr"]]
                print(f"Steps to drop LR: {steps}")
                self.learning_rate = tf.train.piecewise_constant(global_step, steps, cfg.optimizer["lr"])
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    
        if gpus:
            with tf.device('/cpu:0'):
                num_gpus = len(gpus)
                im_split = tf.split(self.in_im, num_gpus, 0)
                ann_split = tf.split(self.in_gt, num_gpus, 0)
            tower_grads, loss_list = [], []
            # Need this variable scope to make sure variables are given names 
            # to enable reuse, probably for build_loss()?
            # It's more like setting up a variable scope just to enable reuse
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                for i, gpu in enumerate(gpus):
                    with tf.device(f"/gpu:{i}"):
                        with tf.name_scope("tower_" + gpu):
                            # These 2 are internal and not exposed
                            input_im, input_ann = im_split[i], ann_split[i]
                            if cfg.num_class == 1:
                                pred = SEResUNet(input_im, cfg, True, 1, 8, "SEResUNet")
                            else:
                                # not supported yet
                                sys.exit()
                            if training:
                                loss = build_loss(pred, input_im, input_ann, cfg)
                                loss_list.append(loss)
                                # return var and its corresponding grad
                                grad = optimizer.compute_gradients(loss)
                                tower_grads.append(grad)
                            # tf.get_variable_scope().reuse_variables()
            if training:         
                mean_grad = average_gradients(tower_grads)
                # Aren't these 2 the same thing? (UPDATE_OPS-applying gradients)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    self.opt_op = optimizer.apply_gradients(mean_grad, global_step=global_step)
                self.loss = tf.add_n(loss_list)
        else:
            if cfg.num_class == 1:
                self.pred = SEResUNet(self.in_im, cfg, True, 1, 8, "SEResUNet")
            else:
                self.pred = SEResUNet(self.in_im, cfg, cfg.num_class + 1, 8, "SEResUNet")
            if training: 
                self.loss = build_loss(self.pred, self.in_im, self.in_gt, cfg) 
                self.opt_op = optimizer.minimize(self.loss, global_step=global_step)

class tf_model_v2():
    def __init__(self, cfg, training=True):
        with tf.device("/cpu:0"):
            self.in_im = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], 1), name="input_im")
            self.in_gt = tf.placeholder(tf.float32, shape=(None, cfg.im_size[0], cfg.im_size[1], 1), name="input_gt")
        model_class = choose_model(cfg.network["name"])
        self.pred = model_class(self.in_im, cfg, training)
        # self.prob = tf.math.sigmoid(self.pred["seg_map"]) 

# Test
if __name__ == "__main__":
    import os
    import numpy as np
    from fw_neutral.utils.config import Config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = Config()
    # cfg.load_from_json("/rdfs/fast/home/sunyingge/pt_ground/configs/UNet_0611.json")
    cfg.load_from_json("/rdfs/fast/home/sunyingge/pt_ground/configs/SEResUNet_0611.json")
    model = tf_model_v2(cfg)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    res_list = sess.run([model.pred["seg_map"]],
        feed_dict={model.in_im: np.random.normal(size=(1, 384, 384, 1)), 
            model.in_gt: np.random.normal(size=(1, 384, 384, 1))})
    for res in res_list:
        print(res.shape)