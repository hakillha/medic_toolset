import pickle, os, sys

import numpy as np
from tensorpack.dataflow import DataFlow, MultiProcessMapData, BatchData

sys.path.insert(0, "../..")
from fw_neutral.utils.data_proc import extra_processing, paths_from_data

class PneuSegDF(DataFlow):
    def __init__(self, args, cfg, test_dir=None):
        super(PneuSegDF, self).__init__()
        self.cfg = cfg
        if not test_dir:
            pdirs, ndirs = paths_from_data(args.train_dir, "pos"), paths_from_data(args.train_dir, "neg")
            if cfg.trainset["pn_ratio"]:
                ndirs = np.random.permutation(ndirs).tolist()[:len(pdirs) // cfg.trainset["pn_ratio"]]
                self.data_dirs = pdirs + ndirs
            else:
                self.data_dirs = pdirs
            self.data_dirs = np.random.permutation(self.data_dirs).tolist()
        else:
            assert os.path.exists(test_dir)
            self.data_dirs = paths_from_data(test_dir, "all")
        self.ex_process = extra_processing(cfg)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.data_dirs[i]

    def __len__(self):
        return len(self.data_dirs)

    def process(self, data_dir):
        data = pickle.load(open(data_dir, "rb"))
        return self.ex_process.preprocess(data["im"], data["mask"])
    
    def prepared(self, num_gpu, batch_size, eval=False):
        ds = MultiProcessMapData(self, max(num_gpu, 8), self.process)
        return BatchData(ds, batch_size)

    def eval_process(self, data_dir):
        data = pickle.load(open(data_dir, "rb"))
        return self.ex_process.preprocess(data["im"], data["mask"], False) + [data_dir]
    
    def eval_prepared(self, num_gpu, batch_size):
        return MultiProcessMapData(self, max(num_gpu, 8), self.eval_process)