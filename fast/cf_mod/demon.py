import argparse, os, sys
import multiprocessing as mp
from functools import partial
from os.path import join as pj

sys.path.insert(0, "/rdfs/fast/home/sunyingge/code/chenfeng")
from misc.utils import get_infos
from misc.data_tools import extract_slice_1ch

def get_slice_dataset(
        data_dir,
        data_dir_post,
        out_dir,
        phase="others",
        link="-",
        pro_fn=None,
        isprint=False
        ):
    info_paths = get_infos(pj(data_dir, data_dir_post), link=link)
    target_dir = pj(out_dir, data_dir_post)
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
    func_1 = partial(extract_slice_1ch,
                    to_dir=out_dir, sub_dir=data_dir_post,
                    min_ct_1ch=-1400, max_ct_1ch=800,
                    pro_fn=None,
                    tofor="disease", phase=phase, isprint=isprint)
    pool = mp.Pool(8)
    pool.map(func_1, info_paths)
    pool.close() 

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_dir",
        default="/rdfs/fast/home/sunyingge/data/COV_19/0508/Train")
    parser.add_argument("--data_dir_postfix", nargs='+', 
        help="""This is added to the ends of both '--data_dir' and
        '--out_dir'.""",
        default=["batch_0505", "batch_0508", "batch_0509", "batch_0510", "batch_0511", 
            "batch_0513", "batch_0514", "batch_0515"])
    parser.add_argument("--out_dir",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0516")
    return parser.parse_args()

args = parse_args()

for post in args.data_dir_postfix:
    get_slice_dataset(args.data_dir, post, args.out_dir)