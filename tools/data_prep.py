import argparse, os, sys
import multiprocessing as mp
from functools import partial
from os.path import join as pj

sys.path.insert(0, ".")
from utils.data_proc import extract_slice

def list_data(data_dir):
    data_list, error_cnt = [], 0
    for root, _, files in os.walk(data_dir):
        pat_id = root.split('/')[-1]
        for f in files:
            ann_file, im_file = None, None
            try:
                if f.endswith("label.nii.gz"):
                    ann_file = pj(root, f)
                    if not f.startswith(pat_id):
                        raise Exception("Anno file inconsistent with subdir!")
                elif f.endswith(".nii.gz"):
                    im_file = pj(root, f)
                    if not f.startswith(pat_id):
                        raise Exception("Img file inconsistent with subdir!")
            except Exception as e:
                print(e)
                print(root)
                print(files)
                error_cnt += 1
            if ann_file and im_file:
                data_list.append([im_file, ann_file])

            # if f.endswith("label.nii.gz"):
            #     ann_file = pj(root, f)
            #     im_file = pj(root, f.split('-')[0] + ".nii.gz")
            #     if os.path.exists(im_file):
            #         data_list.append([im_file, ann_file])
            #         file_cnt += 1
            #     else:
            #         print(im_file)
            #         print('hi')
            #         print(files)
                
    print(f"Number of inconsistencies: {error_cnt}")
    return data_list

def data_prep(
    data_dir,
    out_dir,
    subdir,
    debug=True
):
    data_list = list_data(data_dir)
    # if not os.path.exists(pj(out_dir, subdir)):
    #     os.makedirs(pj(out_dir, subdir))
    # extract = partial(extract_slice,
    #     out_dir=pj(out_dir, subdir), debug=debug)
    # pool = mp.Pool(8)
    # pool.map(extract, data_list)
    # pool.close()

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("-d", "--data_dir",
        default="/rdfs/fast/home/sunyingge/data/COV_19/0508/")
    parser.add_argument("-o", "--out_dir",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0513/")
    # return parser.parse_args()
    # So that this works with jupyter
    return parser.parse_args(args=[
        "-d",
        "/rdfs/fast/home/sunyingge/data/COV_19/0508/",
        "-o",
        "/rdfs/fast/home/sunyingge/data/COV_19/prced_0513/"])

args = parse_args()

for split in ["Train/batch_0505/", "TestSet/", "Val/"]:
    for subset in ["covid_pneu_datasets", "normal_pneu_datasets", "healthy_datasets"]:
        data_prep(pj(args.data_dir, split, subset),
            args.out_dir,
            pj(split, subset))
