import json, os, pickle, sys
from os.path import join as pj

import numpy as np
from tensorpack.dataflow import DataFlow, MultiProcessMapData, BatchData, MapData

sys.path.insert(0, "../..")
from fw_neutral.utils.data_proc import Combine_pndirs, extra_processing, gen_data_list, Patient_quality_filter, paths_from_data
from fw_neutral.utils.dataset import finalize_data_dirs
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
        self.data_dirs = finalize_data_dirs(mode, result_dir, train_dir, testset_dir, cfg)
        self.ex_process = extra_processing(cfg.im_size, cfg.num_class, cfg.preprocess, cfg.loss)
        print(f"\nNumber of samples: {len(self)}\n")
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.data_dirs[i]

    def __len__(self):
        return len(self.data_dirs)
    
    def prepared(self, num_gpu, batch_size, eval=False):
        # use a single process version to debug if needed
        if self.min_num_workers == 0:
            ds = MapData(self, self.ex_process.train_process)
        else:
            ds = MultiProcessMapData(self, max(num_gpu, self.min_num_workers), self.ex_process.train_process)
        return BatchData(ds, batch_size)
    
    def eval_prepared(self, num_gpu, batch_size):
        if self.min_num_workers == 0:
            return MapData(self, self.ex_process.val_process)
        else:
            return MultiProcessMapData(self, max(num_gpu, self.min_num_workers), self.ex_process.val_process)

if __name__ == "__main__":
    import tensorflow as tf
    from fw_neutral.utils.config import Config
    from tqdm import tqdm
    train_dir = "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0617/"
    cfg_file = "/rdfs/fast/home/sunyingge/data/models/workdir_0611/UNet_fpfn_0623_01/config.json"
    cfg = Config()
    cfg.load_from_json(cfg_file)
    # data_dirs, val_dirs = gen_data_list(train_dir, cfg)
    # data_dirs += val_dirs
    data_dirs = json.load(open("/rdfs/fast/home/sunyingge/data/models/workdir_0611/UNet_fpfn_0623_01/result/traindata_dir_list.json", "r"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sess = tf.Session()

    class prc():
        def __init__(self, cfg):
            self.ex_process = extra_processing(cfg.im_size, cfg.num_class, cfg.preprocess, cfg.loss)

        def process(self, data_dir):
            data = pickle.load(open(data_dir, "rb"))
            return self.ex_process.preprocess(data["im"], data["mask"])
            # res = self.ex_process.preprocess(data["im"], data["mask"])
            # return np.array(res)
    ex_prc = prc(cfg)
    
    dataset = tf.data.Dataset.from_generator(lambda: (d for d in data_dirs), output_types=tf.string)
    def readdata(data_dir):
        data = pickle.load(open(data_dir, "rb"))
        return data["im"]
        # return data_dir
    # dataset = dataset.map(lambda data_dir: tf.py_func(readdata, [data_dir], [tf.float32]))
    # dataset = dataset.map(lambda data_dir: tf.py_func(readdata, [data_dir], [tf.string]))
    dataset = dataset.map(lambda data_dir: tf.py_func(ex_prc.process, [data_dir], [tf.float32, tf.bool]))
    dataset = dataset.repeat().batch(8).prefetch(100)
    ds_iter = dataset.make_one_shot_iterator()
    next_ele = ds_iter.get_next()

    l = 10000
    pbar = tqdm(total=l)
    for _ in range(l):
        im = sess.run(next_ele)
        # print((im[0].shape, im[1].shape))
        pbar.update(1)
    pbar.close()