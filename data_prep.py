import argparse, os
import time
from os.path import join as pj

def list_data(data_dir):
    data_list, file_cnt = [], 0
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith("label.nii.gz"):
                ann_file = pj(root, f)
                im_file = pj(root, f.split('_')[0] + ".nii.gz")
                try:
                    data_list.append([im_file, ann_file])
                    file_cnt += 1
                except:
                    # This helps us to find names of data files
                    # that aren't consisent the above rule
                    print(files)
                
    print(f"Number of nii files: {file_cnt}")
    return data_list

def data_prep(
    data_dir,
):
    data_list = list_data(data_dir)

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("-d", "--data_dir",
        default="/rdfs/fast/home/sunyingge/data/COV_19/0508/")
    return parser.parse_args()

args = parse_args()

for split in ["Train/batch_0505/", "TestSet/", "Val/"]:
    for subset in ["covid_pneu_datasets", "normal_pneu_datasets", "healthy_datasets"]:
        data_prep(args.data_dir + split + subset)