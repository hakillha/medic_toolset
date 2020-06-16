import argparse, os, sys
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from os.path import join as pj
from pprint import pprint

import SimpleITK as sitk

# sys.path.insert(0, "/rdfs/fast/home/sunyingge/code/chenfeng")
sys.path.insert(0, "..")
from utils.data_proc import get_infos, extract_slice_single_file

def gen_slice_dataset(data_dir, data_dir_suffix, out_dir, multicat, discard_neg, 
        min_ct_1ch, max_ct_1ch, norm_by_interval, include_healthy, thickness_thres,
        link="-", num_cpu=8):
    info_paths = get_infos(pj(data_dir, data_dir_suffix))

    # extract_slice_sequential(info_paths, out_dir, data_dir_suffix, min_ct_1ch, max_ct_1ch)
    extract = partial(extract_slice_single_file, 
        out_dir=out_dir, data_dir_suffix=data_dir_suffix, min_ct_1ch=min_ct_1ch, max_ct_1ch=max_ct_1ch,
        multicat=multicat, discard_neg=discard_neg, norm_by_interval=norm_by_interval, 
        include_healthy=include_healthy, thickness_thres=thickness_thres)
    pool = mp.Pool(num_cpu)
    stats = defaultdict(int)
    stats_list = pool.map(extract, info_paths)
    pool.close()
    pool.join()
    for im_file, stat in stats_list:
        # if not stat:
        #     print(im_file)
        # else:
        if stat:
            for k, v in stat.items():
                stats[k] += v
    pprint(stats)

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_dir",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/0508/Train",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/0508/Val/"
        # default="/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/0519",
        default="/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/"
        )
    parser.add_argument("--data_dir_suffix", nargs='+', 
        help="""This is added to the ends of both "--data_dir" and "--out_dir".""",
        # default=["batch_0508"],
        # default=["batch_0505", "batch_0507", "batch_0508", "batch_0509", "batch_0510", 
        #     "batch_0511", "batch_0513", "batch_0514", "batch_0515"]
        # default=["batch_0505", "batch_0507", "batch_0508", "batch_0509", "batch_0510", 
        #     "batch_0511", "batch_0513", "batch_0514", "batch_0515", "batch_0516",
        #     "batch_0519", "batch_0520", "batch_0521", "batch_0523", "batch_0527",
        #     "batch_0528", "batch_0529", "batch_0530"]
        default=["batch_0426", "batch_0505"]
        )
    parser.add_argument("--out_dir",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0516",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Test_multicat_0519",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0613",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Test_0616",
    )
    parser.add_argument("--thickness_thres", default=3.0)
    parser.add_argument("--multicat", action="store_true")
    # If both "--include_healthy" and "--discard_neg" enabled,
    # negative samples in sick ppl will be discarded
    parser.add_argument("--discard_neg", action="store_true",
        help="Discard slices that don't contain any lesion.")
    parser.add_argument("--include_healthy", action="store_true")

    # These 2 are deprecated now.
    parser.add_argument("--norm_interval", nargs='+', default=[-1024, 2048]) # default=[-1400, 800]
    parser.add_argument("--norm_by_interval", action="store_true", default=True)
    return parser.parse_args()

# TODO: print out the args for debug purpose
args = parse_args()

for sfx in args.data_dir_suffix:
    gen_slice_dataset(args.data_dir, sfx, args.out_dir, args.multicat, args.discard_neg, 
        args.norm_interval[0], args.norm_interval[1], args.norm_by_interval, args.include_healthy,
        args.thickness_thres)