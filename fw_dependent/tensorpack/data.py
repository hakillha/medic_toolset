import pickle, sys

import numpy as np
from tensorpack.dataflow import DataFlow

sys.path.insert(0, "../..")
from fw_neutral.utils.data_proc import extra_processing, paths_from_data

class PneuSegDF(DataFlow):
    def __init__(self, args, cfg):
        super(PneuSegDF, self).__init__()
        self.cfg = cfg
        pdirs, ndirs = paths_from_data(args.train_dir, "pos"), paths_from_data(args.train_dir, "neg")
        if cfg.trainset["pn_ratio"]:
            ndirs = np.random.permutation(ndirs).tolist()[:len(pdirs) // cfg.trainset["pn_ratio"]]
            self.data_dirs = pdirs + ndirs
        else:
            self.data_dirs = pdirs
        self.data_dirs = np.random.permutation(self.data_dirs).tolist()
        self.ex_process = extra_processing(cfg)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.data_dirs[i]

    def __len__(self):
        return len(self.data_dirs)

    def process(self, data_dir):
        data = pickle.load(open(data_dir, "rb"))
        return self.ex_process.preprocess(data["im"], data["mask"])