import tensorflow as tf
from tensorpack.train import ModelDesc
from tensorpack.tfutils import get_global_step_var

import sys
sys.path.insert(0, "../..")
from fw_dependent.tf.model.ASEUNet import SEResUNet
from fw_dependent.tf.model.UNet import UNet
from fw_dependent.tf.model.tf_layers import build_loss

# MODEL_MAP = {
#     "SEResUNet": SEResUNet,
#     "UNet": UNet
# }

class Tensorpack_model(ModelDesc):
    def __init__(self, cfg):
        super(Tensorpack_model, self).__init__()
        self.cfg = cfg

    def inputs(self):
        return [tf.TensorSpec([None, self.cfg.im_size[0], self.cfg.im_size[1], 1], name="input_im"),
            tf.TensorSpec([None, self.cfg.im_size[0], self.cfg.im_size[1], 1], name="input_gt")]
    
    def build_graph(self, im, gt):
        # network = MODEL_MAP[self.cfg.network["name"]]
        # pred = network(im, self.cfg)
        if self.cfg.network["name"] == "SEResUNet":
            pred = SEResUNet(im, self.cfg)
        elif self.cfg.network["name"] == "UNet":
            pred = UNet(im, self.cfg)
        loss = build_loss(pred, im, gt, self.cfg)
        tf.summary.scalar("Loss", loss)
        return loss
    
    def optimizer(self):
        learning_rate = tf.get_variable("learning_rate", 
            initializer=self.cfg.optimizer["lr"][0], trainable=False)
        return tf.train.AdamOptimizer(learning_rate)

# if __name__ == "__main__":
#     import os, sys
#     sys.path.insert(0, "../..")
#     from fw_neutral.utils.config import Config
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     cfg = Config()
#     cfg.load_from_json("/rdfs/fast/home/sunyingge/pt_ground/configs/UNet_0611.json")
#     model = Tensorpack_model(cfg)
#     im, gt = model.inputs()
#     loss = model.build_graph(im, gt)