#!/usr/bin/env python
# coding: utf-8
import argparse, os, sys
def get_abspath(path):
    path_ = os.path.expanduser(path)
    return os.path.abspath(path_)

import tensorflow as tf
import keras

from misc.utils import keras_to_graphdef, tf_to_graphdef

def parse_args():
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("framework", choices=["tensorflow", "keras"], default="tensorflow")
    parser.add_argument("out_dir",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/inf/")
    parser.add_argument("--version")
    parser.add_argument("--output_filename", default="model.graphdef")

    # Keras
    parser.add_argument("--model")
    parser.add_argument("--weights_file")
    
    # TF
    parser.add_argument("--meta_file",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_7.ckpt.meta",)
    parser.add_argument("--ckpt_file",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_7.ckpt",)

    return parser.parse_args()

if __name__ == "__main__":
    # argv = sys.argv 
    # assert len(argv) == 5, "We Need FIVE Parameters For The Script"  "\n==> python script.py frame network weight_file version" 
    args = parse_args()
    frame, network, weight_file, version, to_dir = args.framework, args.model, args.weights_file, args.version, args.out_dir
    # assert frame in ["tensorflow", "keras"], "Please Specify The Right CNN Frame"

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95


    if frame == "tensorflow":
        ## convert tensorflow graph to graphdef
        # meta_file = "/rdfs/fast/home/chenfeng/Documents/Models/LungPneumonia/MULTI_TF_SERESUNET_PNEU_0.00050_2020031013/best_model-33.meta"
        # ckpt_file = "/rdfs/fast/home/chenfeng/Documents/Models/LungPneumonia/MULTI_TF_SERESUNET_PNEU_0.00050_2020031013/best_model-33"
        
        tf_to_graphdef(
            args.meta_file, args.ckpt_file, to_dir, config, 
            # output_names=["tower_3/SEResUNet/conv_final/BiasAdd"], 
            # output_names=["SEResUNet/conv_final"],
            output_names=["SEResUNet/conv_final/BiasAdd",],
            graphdef_name=args.output_filename)
    elif frame == "keras":
        ## convert keras model to graphdef
        if network == "hrnet":  
            sys.path.insert(0, get_abspath("~/code/chenfeng/segmenter/"))
            from hrnet import HighResolutionNet as network_fn
            args_dict = {
                    "input_size" : (256, 256, 1),
                    "num_classes": 2,
                    }
        elif network == "unet3d":
            sys.path.insert(0, get_abspath("~/NetWork/keras/segmenter/"))
            from unet3d_gn import Unet3d as network_fn
            args_dict = {
                    "input_size" : (4, 256, 256, 2),
                    "k_size": 3,
                    "dilation_rate": (1, 2, 2)
                    }        
        else:
            try:
                sys.exit()
            except:
                os._exit()
        weight_file = get_abspath(weight_file) 
        keras_to_graphdef(network_fn, weight_file, 
                            to_dir, config, args.output_filename, 
                            **args_dict,  
                )
