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
    
    def build_graph(self, im, gt):
        model_class = choose_model(self.cfg)
        self.pred = model_class(im, self.cfg, get_current_tower_context().is_training)
        self.loss = build_loss(self.pred, im, gt, self.cfg)
        tf.summary.scalar("Loss", self.loss)
        return self.loss
    
    def optimizer(self):
        learning_rate = tf.get_variable("learning_rate", 
            initializer=self.cfg.optimizer["lr"][0], trainable=False)
        tf.summary.scalar("LR", learning_rate)
        return tf.train.AdamOptimizer(learning_rate)
    
    def build_inf_graph(self):
        self.in_im = tf.placeholder(tf.float32, 
            shape=(None, self.cfg.im_size[0], self.cfg.im_size[1], 1), name=self.in_im_tname)
        self.in_gt = tf.placeholder(tf.float32, 
            shape=(None, self.cfg.im_size[0], self.cfg.im_size[1], 1), name=self.in_gt_tname)
        with TowerContext("", is_training=False):
            self.build_graph(self.in_im, self.in_gt)

if __name__ == "__main__":
    import os
    import numpy as np
    from fw_neutral.utils.config import Config
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cfg = Config()
    cfg.load_from_json("/rdfs/fast/home/sunyingge/pt_ground/configs/UNet_0611.json")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    test_model = Tensorpack_model(cfg, True)
    test_model.build_inf_graph()
    sess.run(tf.global_variables_initializer())
    res_list = sess.run([test_model.pred["seg_map"]],
        feed_dict={test_model.in_im: np.random.normal(size=(1, 384, 384, 1)), 
            test_model.in_gt: np.random.normal(size=(1, 384, 384, 1))})
    for res in res_list:
        print(res.shape)
