import tensorflow as tf
from tensorpack.train import ModelDesc
from tensorpack.tfutils import get_global_step_var, get_current_tower_context

import sys
sys.path.insert(0, "../..")
from fw_dependent.tf.model.tf_layers import choose_model, build_loss

class Tensorpack_model(ModelDesc):
    def __init__(self, cfg, training=True):
        super(Tensorpack_model, self).__init__()
        self.cfg = cfg
        # self.training_network = training

    def inputs(self):
        return [tf.TensorSpec([None, self.cfg.im_size[0], self.cfg.im_size[1], 1], name="input_im"),
            tf.TensorSpec([None, self.cfg.im_size[0], self.cfg.im_size[1], 1], name="input_gt")]
    
    def build_graph(self, im, gt):
        model_class = choose_model(self.cfg)
        # pred = model_class(im, self.cfg, self.training_network)
        pred = model_class(im, self.cfg, get_current_tower_context().is_training)
        loss = build_loss(pred, im, gt, self.cfg)
        tf.summary.scalar("Loss", loss)
        return loss
    
    def optimizer(self):
        learning_rate = tf.get_variable("learning_rate", 
            initializer=self.cfg.optimizer["lr"][0], trainable=False)
        return tf.train.AdamOptimizer(learning_rate)

if __name__ == "__main__":
    import os
    from fw_neutral.utils.config import Config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = Config()
    cfg.load_from_json("/rdfs/fast/home/sunyingge/pt_ground/configs/UNet_0611.json")
    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95
    # sess = tf.Session(config=config)
    # model = MODEL_MAP[cfg.network["name"]](cfg)
    train_model = Tensorpack_model(cfg, True)
