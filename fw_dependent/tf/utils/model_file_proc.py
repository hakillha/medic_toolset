import os, sys
import tensorflow as tf

sys.path.insert(0, "../../..")
import fw_dependent.tf.model.tf_layers as tf_layer_seresunet
from fw_neutral.utils.config import Config

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    cfg = Config()
    # cfg_f = "/rdfs/fast/home/sunyingge/data/models/workdir_0522/Finetune_0528/cfg.json"
    cfg_f = "/rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_0611_1412_05/base_0526.json"
    cfg.load_from_json(cfg_f)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model = tf_layer_seresunet.tf_model(cfg, False)
    saver = tf.train.Saver()
    # f = "/rdfs/fast/home/sunyingge/data/models/workdir_0522/Finetune_0528/epoch_2.ckpt"
    f = "/rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_0611_1412_05/model-12058"
    saver.restore(sess, f)