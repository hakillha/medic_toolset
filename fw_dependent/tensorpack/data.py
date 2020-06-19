import json, os, pickle, sys

import numpy as np
from tensorpack.dataflow import DataFlow, MultiProcessMapData, BatchData

sys.path.insert(0, "../..")
from fw_neutral.utils.data_proc import extra_processing, Patient_quality_filter, paths_from_data

class PneuSegDF(DataFlow):
    def __init__(self, args, cfg):
        super(PneuSegDF, self).__init__()
        self.cfg = cfg
        self.min_num_workers = args.min_num_workers

        # Put this section of code into a seperate data class?
        if args.mode == "val":
            self.data_dirs = paths_from_data(args.valset_dir, None, cfg.valset["pos_or_neg"])
            if len(cfg.valset["patient_filter_file"]):
                self.data_dirs = Patient_quality_filter(self.data_dirs, 
                    cfg.valset["md5_map"], cfg.valset["patient_filter_file"], cfg.valset["quality"])
        elif args.mode == "sess_eval":
            self.data_dirs = paths_from_data(args.testset_dir, None, "all")
        else: # training
            if len(cfg.trainset["data_list"]):
                datalist = json.load(open(cfg.trainset["data_list"], "r"))
                pdirs, ndirs = paths_from_data(None, datalist, "pos"), paths_from_data(None, datalist, "neg")
            else:
                pdirs, ndirs = paths_from_data(args.train_dir, None, "pos"), paths_from_data(args.train_dir, None, "neg")
                if len(cfg.trainset["patient_filter_file"]):
                    pdirs = Patient_quality_filter(pdirs, cfg.trainset["md5_map"], cfg.trainset["patient_filter_file"], cfg.trainset["quality"])
                    ndirs = Patient_quality_filter(ndirs, cfg.trainset["md5_map"], cfg.trainset["patient_filter_file"], cfg.trainset["quality"])

            if cfg.trainset["pn_ratio"]:
                if cfg.trainset["pn_ratio"] == "all":
                    ndirs = np.random.permutation(ndirs).tolist()
                else:
                    ndirs = np.random.permutation(ndirs).tolist()[:len(pdirs) // cfg.trainset["pn_ratio"]]
                self.data_dirs = pdirs + ndirs
            else:
                self.data_dirs = pdirs

            self.data_dirs = np.random.permutation(self.data_dirs).tolist() 
            
        
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
        ds = MultiProcessMapData(self, max(num_gpu, self.min_num_workers), self.process)
        return BatchData(ds, batch_size)

    def eval_process(self, data_dir):
        data = pickle.load(open(data_dir, "rb"))
        return self.ex_process.preprocess(data["im"], data["mask"], False) + [data_dir, data["im"]]
    
    def eval_prepared(self, num_gpu, batch_size):
        return MultiProcessMapData(self, max(num_gpu, self.min_num_workers), self.eval_process)