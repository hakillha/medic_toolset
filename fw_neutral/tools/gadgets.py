import argparse, json, os, shutil
from os.path import join as pj

import sys
sys.path.insert(0, "../..")
from fw_neutral.utils.metrics import Evaluation, Patient
from fw_neutral.utils.data_proc import get_infos, Patient_quality_filter

def check_walk():
    patient_set = set()

    data_dir = "/rdfs/fast/home/sunyingge/data/COV_19/0508/Train"
    # data_dir = "/rdfs/data/pneuDataset/SegPneu/Train"
    # data_dir = "/rdfs/fast/home/sunyingge/data/COV_19/0508/Train_raw_0617"
    data_dir_suffix_list = ["batch_0505", "batch_0507", "batch_0508", "batch_0509", "batch_0510", 
        "batch_0511", "batch_0513", "batch_0514", "batch_0515", "batch_0516",
        "batch_0519", "batch_0520", "batch_0521", "batch_0523", "batch_0527",
        "batch_0528", "batch_0529", "batch_0530"]
    info_dirs = []
    for suffix in data_dir_suffix_list:
        info_dirs += get_infos(pj(data_dir, suffix))
    for info_dir in info_dirs:
        p_id = Patient.find_id(info_dir[0])
        patient_set.add(p_id)
    info_dirs = [info_dir[0] for info_dir in info_dirs]
    Patient_quality_filter(info_dirs, "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/MD5_map.csv", "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/trainq.csv")
    
    print(len(patient_set))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["parse_res_pkl", "check_walk", "assign_id", "other"])
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "parse_res_pkl":
        evaluation = Evaluation()
        evaluation.pixel_wise_result(args.in_dir)
    elif args.mode == "check_walk":
        check_walk()
    elif args.mode == "assign_id":
        data_dirs = get_infos(args.in_dir)
        p_id = 1
        for data_dir in data_dirs:
            target_dir = pj(args.out_dir, "healthy", f"{p_id:032}")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for f in data_dir:
                shutil.copy(f, target_dir)
            p_id += 1
    else:
        bad_dir_list = json.load(open("/rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_0617_1203_00/model-118140_res_bad_slice.json", "r"))
        pos, neg = 0, 0
        for bad_dir in bad_dir_list:
            if "pos" in bad_dir:
                pos += 1
            else:
                neg += 1
        print((pos, neg, pos + neg))