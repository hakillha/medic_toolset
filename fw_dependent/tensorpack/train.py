import argparse, os, shutil, time
import multiprocessing as mp
from os.path import join as pj

from tensorpack.dataflow import MultiProcessMapData, BatchData
# from tensorpack.dataflow import MultiProcessMapAndBatchData
from tensorpack.train import TrainConfig, launch_train_with_config, SyncMultiGPUTrainerParameterServer
from tensorpack.train import SyncMultiGPUTrainerReplicated
from tensorpack.tfutils import SmartInit
from tensorpack.callbacks import ModelSaver, ScheduledHyperParamSetter, GPUUtilizationTracker, PeriodicCallback
from tensorpack.callbacks import MergeAllSummaries, ScalarPrinter, TFEventWriter, JSONWriter
from tensorpack.utils.gpu import get_num_gpu
from tensorpack import QueueInput

import sys
sys.path.insert(0, "../..")
from data import PneuSegDF
from fw_neutral.utils.config import Config
from tp_model import Tensorpack_model

# from tensorpack.dataflow import TestDataSpeed, PrintData

def parse_args():
    parser = argparse.ArgumentParser("""""")
    # parser.add_argument("mode", choices=["train"])
    parser.add_argument("config", help="Config file.")
    parser.add_argument("--gpus_to_use", nargs='+')
    
    parser.add_argument("-o", "--output_dir",
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_")
    parser.add_argument("--train_dir", help="Training set directory.",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0608/"
        )
    parser.add_argument("--resume")

    return parser.parse_args()

def train(args, cfg):
    df = PneuSegDF(args, cfg)
    print(f"\nNumber of training samples: {len(df)}\n")
    # ds = MultiProcessMapData(df, 8, df.process, buffer_size=512)
    ds = MultiProcessMapData(df, num_gpu, df.process)
    ds = BatchData(ds, cfg.batch_size)
    schedule = [(ep + 1, lr / num_gpu) for ep, lr in zip([0] + cfg.optimizer["epoch_to_drop_lr"], cfg.optimizer["lr"])]
    train_cfg = TrainConfig(
        model=Tensorpack_model(cfg),
        data=QueueInput(ds),
        steps_per_epoch=len(ds) // num_gpu + 1,
        callbacks=[
            PeriodicCallback(
                ModelSaver(keep_checkpoint_every_n_hours=1, checkpoint_dir=output_dir), 
                every_k_epochs=1),
            ScheduledHyperParamSetter("learning_rate", schedule),
            GPUUtilizationTracker(),
            MergeAllSummaries(1),
            ],
        monitors=[
            ScalarPrinter(True, whitelist=["Loss"]),
            # ScalarPrinter(True),
            TFEventWriter(), 
            JSONWriter()
        ],
        max_epoch=cfg.max_epoch,
        session_init=SmartInit(args.resume)
    )
    output_dir = args.output_dir + time.strftime("%m%d_%H%M_%S", time.localtime())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Avoid overriding config file
    assert not os.path.exists(pj(output_dir, args.config))
    shutil.copy(args.config, output_dir)
    launch_train_with_config(train_cfg, SyncMultiGPUTrainerReplicated(num_gpu))

if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus_to_use)
    cfg = Config()
    cfg.load_from_json(args.config)
    num_gpu = max(get_num_gpu(), 1)
    
