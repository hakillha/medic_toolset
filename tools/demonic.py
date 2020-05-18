import argparse, os, sys
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from os.path import join as pj

import SimpleITK as sitk

# sys.path.insert(0, "/rdfs/fast/home/sunyingge/code/chenfeng")
sys.path.insert(0, "..")
from utils.data_proc import get_infos, extract_slice_single_file

def gen_slice_dataset(data_dir, data_dir_post, out_dir, link="-", 
                      min_ct_1ch=-1400, max_ct_1ch=800, num_cpu=8):
    info_paths = get_infos(pj(data_dir, data_dir_post), link=link)

    # extract_slice_sequential(info_paths, out_dir, data_dir_post, min_ct_1ch, max_ct_1ch)
    extract = partial(extract_slice_single_file, 
        out_dir=out_dir, data_dir_post=data_dir_post, min_ct_1ch=-1400, max_ct_1ch=800)
    pool = mp.Pool(num_cpu)
    pool.map(extract, info_paths)
    pool.close()

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_dir",
        default="/rdfs/fast/home/sunyingge/data/COV_19/0508/Train",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/0508/Val/"
        )
    parser.add_argument("--data_dir_postfix", nargs='+', 
        help="""This is added to the ends of both '--data_dir' and
        '--out_dir'.""",
        # default=["batch_0513", "batch_0514", "batch_0515"],
        # default=["batch_0505", "batch_0507", "batch_0508", "batch_0509", "batch_0510", 
        #     "batch_0511", "batch_0513", "batch_0514", "batch_0515"]
        default=["batch_0505", "batch_0508", "batch_0509", "batch_0510", "batch_0511", 
            "batch_0513", "batch_0514", "batch_0515"]
            )
    parser.add_argument("--out_dir",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0516",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0518_fixed",
    )
    parser.add_argument("--thickness_thres", default=3.0)
    return parser.parse_args()

args = parse_args()

for post in args.data_dir_postfix:
    gen_slice_dataset(args.data_dir, post, args.out_dir)