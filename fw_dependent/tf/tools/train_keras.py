import argparse, os, sys
import tensorflow as tf
tf.executing_eagerly()

sys.path.insert(0, "../../..")
from fw_dependent.tf.model.ASEUNet_keras import InputLayer
from fw_neutral.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("config",
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/test01/cfg02.json")
    parser.add_argument("--gpu", nargs='+')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu)
    cfg = Config()
    cfg.load_from_json(args.config)

    model = InputLayer(cfg)
    print(model(tf.random.normal([2, 384, 384, 1])).shape)