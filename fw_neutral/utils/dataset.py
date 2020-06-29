import json, os, sys
from os.path import join as pj

sys.path.insert(0, "../..")
from fw_neutral.utils.data_proc import gen_data_list, Patient_quality_filter, paths_from_data
from fw_neutral.utils.metrics import Pneu_type

def finalize_data_dirs(mode, result_dir, train_dir, testset_dir, cfg):
    val_dirs = None
    if mode == "train": # training
        # the resuming train/val list is prioritized over all other settings
        if os.path.exists(pj(result_dir, "traindata_dir_list.json")):
            data_dirs = json.load(open(pj(result_dir, "traindata_dir_list.json"), "r"))
            val_dirs = json.load(open(pj(result_dir, "valdata_dir_list.json"), "r"))
        else:
            data_dirs, val_dirs = gen_data_list(train_dir, cfg)
            json.dump(data_dirs, open(pj(result_dir, "traindata_dir_list.json"), "w"))
            json.dump(val_dirs, open(pj(result_dir, "valdata_dir_list.json"), "w"))
    elif mode == "val":
        data_dirs = json.load(open(cfg.valset["datalist_dir"], "r"))
    elif mode == "test":
        data_dirs = paths_from_data(testset_dir, None, "all")
        if not cfg.testset["include_healthy"]:
            data_dirs = [data_dir for data_dir in data_dirs if Pneu_type(data_dir, True) != "healthy"]
    elif mode == "trainset_eval":
        data_dirs = paths_from_data(cfg.trainset_eval["root_dir"], None, cfg.trainset_eval["pos_or_neg"])
        if len(cfg.trainset_eval["patient_filter_file"]):
            data_dirs, _ = Patient_quality_filter(data_dirs, cfg.trainset_eval["md5_map"], 
                cfg.trainset_eval["patient_filter_file"], cfg.trainset_eval["quality"])
    return data_dirs, val_dirs