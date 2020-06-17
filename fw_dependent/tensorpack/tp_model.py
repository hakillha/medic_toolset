from pprint import pprint

import tensorflow as tf
from tensorpack.train import ModelDesc
from tensorpack.tfutils import get_global_step_var, get_current_tower_context, TowerContext

import sys
sys.path.insert(0, "../..")
from fw_dependent.tf.model.tf_layers import choose_model, build_loss

class Tensorpack_model(ModelDesc):
    def __init__(self, cfg, training=True):
        super(Tensorpack_model, self).__init__()
        self.cfg = cfg
        # self.training_network = training
        self.in_im_tname = "input_im"
        self.in_gt_tname = "input_gt"

    def inputs(self):
        return [tf.TensorSpec([None, self.cfg.im_size[0], self.cfg.im_size[1], 1], name=self.in_im_tname),
            tf.TensorSpec([None, self.cfg.im_size[0], self.cfg.im_size[1], 1], name=self.in_gt_tname)]
    
    def build_graph(self, im, gt, training=None):
        model_class = choose_model(self.cfg)
        if training == None:
            training = get_current_tower_context().is_training
        self.ops = model_class(im, self.cfg, training)
        self.loss = build_loss(self.ops, im, gt, self.cfg)
        tf.summary.scalar("Loss", self.loss)
        return self.loss
    
    def optimizer(self):
        learning_rate = tf.get_variable("learning_rate", 
            initializer=self.cfg.optimizer["lr"][0], trainable=False)
        tf.summary.scalar("LR", learning_rate)
        return tf.train.AdamOptimizer(learning_rate)
    
    def build_inf_graph(self, training):
        self.in_im = tf.placeholder(tf.float32, 
            shape=(None, self.cfg.im_size[0], self.cfg.im_size[1], 1), name=self.in_im_tname)
        self.in_gt = tf.placeholder(tf.float32, 
            shape=(None, self.cfg.im_size[0], self.cfg.im_size[1], 1), name=self.in_gt_tname)
        with TowerContext("", is_training=training):
            self.build_graph(self.in_im, self.in_gt)

if __name__ == "__main__":
    import os
    import numpy as np
    from tqdm import tqdm
    from fw_neutral.utils.config import Config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = Config()
    # cfg_file = "/rdfs/fast/home/sunyingge/pt_ground/configs/UNet_0611.json"
    # cfg_file = "/rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_0615_1926_49/UNet_0615.json"
    cfg_file = "/rdfs/fast/home/sunyingge/pt_ground/configs/UNet_0615.json"
    cfg.load_from_json(cfg_file)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    test_model = Tensorpack_model(cfg, True)
    # Test both training and inference graphs
    test_model.build_inf_graph(False)
    optimizer = test_model.optimizer()
    if cfg.network["norm_layer"] == "BN_layers":
        with tf.control_dependencies(update_ops):
            opt_op = optimizer.minimize(test_model.loss)
    else:
        opt_op = optimizer.minimize(test_model.loss)
    # opt_op = optimizer.minimize(test_model.loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print("\ntf.layers.BN update operations:")
    pprint(update_ops)

    sess.run(tf.global_variables_initializer())
    steps = 10000
    pbar = tqdm(total=steps)
    for _ in range(steps):
        # fetches = [test_model.ops["seg_map"]]
        fetches = [opt_op]
        res_list = sess.run(fetches,
            feed_dict={test_model.in_im: np.random.normal(size=(4, 384, 384, 1)), 
                test_model.in_gt: np.random.normal(size=(4, 384, 384, 1))})
        pbar.update(1)
    pbar.close()
    for res in res_list:
        print(res.shape)
