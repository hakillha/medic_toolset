import json, os, pickle, sys
from os.path import join as pj

import numpy as np
from tensorpack.dataflow import DataFlow, MultiProcessMapData, BatchData, MapData

sys.path.insert(0, "../..")
from fw_neutral.utils.data_proc import Combine_pndirs, extra_processing, gen_data_list, Patient_quality_filter, paths_from_data
from fw_neutral.utils.metrics import Pneu_type

class PneuSegDF(DataFlow):
    def __init__(self, mode, result_dir, train_dir, testset_dir, min_num_workers, cfg):
        """
            args:
                result_dir: Set to None to non-training modes.
        """
        super(PneuSegDF, self).__init__()
        self.cfg = cfg
        self.min_num_workers = min_num_workers

        if mode == "train": # training
            # the resuming train/val list is prioritized over all other settings
            if os.path.exists(pj(result_dir, "traindata_dir_list.json")):
                self.data_dirs = json.load(open(pj(result_dir, "traindata_dir_list.json"), "r"))
            else:
                # if len(cfg.trainset["data_list"]):
                #     datalist = json.load(open(cfg.trainset["data_list"], "r"))
                #     pdirs, ndirs = paths_from_data(None, datalist, "pos"), paths_from_data(None, datalist, "neg")
                # else:
                #     pdirs, ndirs = paths_from_data(train_dir, None, "pos"), paths_from_data(train_dir, None, "neg")
                #     if len(cfg.trainset["patient_filter_file"]):
                #         pdirs, _ = Patient_quality_filter(pdirs, cfg.trainset["md5_map"], cfg.trainset["patient_filter_file"], cfg.trainset["quality"])
                #         ndirs, _ = Patient_quality_filter(ndirs, cfg.trainset["md5_map"], cfg.trainset["patient_filter_file"], cfg.trainset["quality"])
                # self.data_dirs = Combine_pndirs(cfg.trainset["pn_ratio"], pdirs, ndirs)
                # self.data_dirs = np.random.permutation(self.data_dirs).tolist()
                # val_dirs = self.data_dirs[-int(len(self.data_dirs) * cfg.trainset["val_ratio"]):]
                # self.data_dirs = self.data_dirs[:len(self.data_dirs) - len(val_dirs)]
                self.data_dirs, val_dirs = gen_data_list(train_dir, cfg)
                json.dump(self.data_dirs, open(pj(result_dir, "traindata_dir_list.json"), "w"))
                json.dump(val_dirs, open(pj(result_dir, "valdata_dir_list.json"), "w"))
        elif mode == "val":
            self.data_dirs = json.load(open(cfg.valset["datalist_dir"], "r"))
        elif mode == "test":
            data_dirs = paths_from_data(testset_dir, None, "all")
            if cfg.testset["include_healthy"]:
                self.data_dirs = data_dirs
            else:
                self.data_dirs = [data_dir for data_dir in data_dirs if Pneu_type(data_dir, True) != "healthy"]
        elif mode == "trainset_eval":
            self.data_dirs = paths_from_data(cfg.trainset_eval["root_dir"], None, cfg.trainset_eval["pos_or_neg"])
            if len(cfg.trainset_eval["patient_filter_file"]):
                self.data_dirs, _ = Patient_quality_filter(self.data_dirs, 
                    cfg.trainset_eval["md5_map"], cfg.trainset_eval["patient_filter_file"], cfg.trainset_eval["quality"])
        
        self.ex_process = extra_processing(cfg.im_size, cfg.num_class, cfg.preprocess, cfg.loss)
        print(f"\nNumber of samples: {len(self)}\n")
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.data_dirs[i]

    def __len__(self):
        return len(self.data_dirs)

    def process(self, data_dir):
        data = pickle.load(open(data_dir, "rb"))
        return self.ex_process.preprocess(data["im"], data["mask"])
    
    def prepared(self, num_gpu, batch_size, eval=False):
        # use a single process version to debug if needed
        if self.min_num_workers == 0:
            ds = MapData(self, self.process)
        else:
            ds = MultiProcessMapData(self, max(num_gpu, self.min_num_workers), self.process)
        return BatchData(ds, batch_size)

    def eval_process(self, data_dir):
        data = pickle.load(open(data_dir, "rb"))
        return self.ex_process.preprocess(data["im"], data["mask"], False) + [data_dir, data["im"]]
    
    def eval_prepared(self, num_gpu, batch_size):
        if self.min_num_workers == 0:
            return MapData(self, self.eval_process)
        else:
            return MultiProcessMapData(self, max(num_gpu, self.min_num_workers), self.eval_process)