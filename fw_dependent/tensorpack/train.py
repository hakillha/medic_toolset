import argparse, os, shutil, time
import multiprocessing as mp
from os.path import join as pj

import tensorflow as tf

from tensorpack.dataflow import MultiProcessMapData, BatchData
# from tensorpack.dataflow import MultiProcessMapAndBatchData
from tensorpack.train import TrainConfig, launch_train_with_config, SyncMultiGPUTrainerParameterServer
from tensorpack.train import SyncMultiGPUTrainerReplicated, SimpleTrainer
from tensorpack.tfutils import SmartInit
from tensorpack.callbacks import ModelSaver, ScheduledHyperParamSetter, GPUUtilizationTracker, PeriodicCallback
from tensorpack.callbacks import MergeAllSummaries, ScalarPrinter, TFEventWriter, JSONWriter, ProgressBar
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu
from tensorpack import QueueInput

import sys
sys.path.insert(0, "../..")
from data import PneuSegDF
from tp_model import Tensorpack_model
from fw_dependent.tf.model.tf_layers import tf_model_v2
from fw_dependent.tf.tools.train_tf_lowlevel import evaluation
from fw_neutral.utils.config import Config

# from tensorpack.dataflow import TestDataSpeed, PrintData

def parse_args():
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("mode", choices=["train", "sess_eval"])
    parser.add_argument("config", help="Config file.")
    parser.add_argument("--gpus_to_use", nargs='+')
    parser.add_argument("-o", "--output_dir",
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_")
    parser.add_argument("--train_dir", help="Training set directory.",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0608/"
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0613/"
        )
    parser.add_argument("--train_debug", action="store_true")

    parser.add_argument("--resume", default="")
    parser.add_argument("--resume_epoch", help="Checkpoint epoch plus one.", default=1)

    # Eval mode related
    parser.add_argument("--testset_dir", nargs='+',
        default=["/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/0519/normal_pneu_datasets",
        "/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/0519/covid_pneu_datasets"]
        # default=["/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/healthy_datasets"]
        )
    parser.add_argument("--batch_size", type=int,
        help="Provided here to enable easy overwritting (particularly useful for evaluation).")
    parser.add_argument("--model_file",
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_0612_1451_14/model-4823"
        )
    parser.add_argument("--eval_multi", action="store_true")
    parser.add_argument("--model_folder")
    parser.add_argument("--model_list", nargs='+')

    return parser.parse_args()

def train(args, cfg):
    df = PneuSegDF(args, cfg)
    num_gpu = max(get_num_gpu(), 1)
    print(f"\nNumber of training samples: {len(df)}\n")
    # ds = MultiProcessMapData(df, 8, df.process, buffer_size=512)
    ds = MultiProcessMapData(df, num_gpu, df.process)
    ds = BatchData(ds, cfg.batch_size)
    schedule = [(ep + 1, lr / num_gpu) for ep, lr in zip([0] + cfg.optimizer["epoch_to_drop_lr"], cfg.optimizer["lr"])]
    if os.path.exists(os.path.dirname(args.resume)):
        assert args.resume_epoch != 1
        output_dir = os.path.dirname(args.resume)
    else:
        output_dir = args.output_dir + time.strftime("%m%d_%H%M_%S", time.localtime())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Avoid overriding config file
    if os.path.exists(pj(output_dir, os.path.basename(args.config))):
        input("Config file will NOT be overwritten. Press Enter to continue...")
    else:
        shutil.copy(args.config, output_dir)
    logger.set_logger_dir(pj(output_dir, "log"))
    train_cfg = TrainConfig(
        model=Tensorpack_model(cfg),
        data=QueueInput(ds),
        steps_per_epoch=len(ds) // num_gpu + 1,
        callbacks=[
            # PeriodicCallback overwritten the frequency of what's wrapped
            PeriodicCallback(ModelSaver(50, checkpoint_dir=output_dir), every_k_epochs=1),
            ScheduledHyperParamSetter("learning_rate", schedule),
            GPUUtilizationTracker(),
            MergeAllSummaries(1 if args.train_debug else 0),
            # ProgressBar(["Loss"])
            ],
        monitors=[
            # ScalarPrinter(True, whitelist=["Loss", "LR"]),
            ScalarPrinter(True),
            # ScalarPrinter(),
            TFEventWriter(), 
            # JSONWriter()
            ],
        max_epoch=cfg.max_epoch,
        session_init=SmartInit(args.resume),
        starting_epoch=args.resume_epoch
    )
    launch_train_with_config(train_cfg, 
        SyncMultiGPUTrainerReplicated(num_gpu) if num_gpu > 1 else SimpleTrainer())

if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus_to_use)
    cfg = Config()
    cfg.load_from_json(args.config)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.mode == "train":
        train(args, cfg)
    elif args.mode == "sess_eval":
        args.thickness_thres = 3.0
        args.eval_debug = False
        args.viz = False
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        sess = tf.Session(config=config)
        # Change this to use towercontext-model.build_graph(placeholders) for consistency
        # Refer to tp doc inference section for details
        # model = tf_model_v2(cfg, False)
        model = Tensorpack_model(cfg)
        model.build_inf_graph()
        if args.eval_multi:
            for mname in args.model_list:
                args.model_file = pj(args.model_folder, mname)
                args.pkl_dir = args.model_file + "_res.pkl"
                evaluation("eval_multi", sess, args, cfg, model)
                print(f"\nFinished evaluating {mname}.\n")
        else:
            args.pkl_dir = args.model_file + "_res.pkl"
            evaluation("eval_multi", sess, args, cfg, model)