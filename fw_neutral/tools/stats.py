import argparse, os, sys
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, "../..")
from fast.cf_mod.misc.utils import get_infos
from fw_neutral.utils.metrics import Evaluation

# def size_stats(data_paths):
#     # 
#     for im_file, anno_file, _ in data_paths:
#         anno_sitk = sitk.ReadImage(anno_file, sitk.sitkInt16)

def parse_args():
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("--testset_dir", 
        default=["/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/0519/normal_pneu_datasets",
        "/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/0519/covid_pneu_datasets"]
        # default=[]
        )
    parser.add_argument("--output_dir", 
        default="/rdfs/fast/home/sunyingge/data/misc")
    parser.add_argument("--res_pkl_file", type=str,
        default="/rdfs/fast/home/sunyingge/data/misc/set_stats.pkl"
        )
    parser.add_argument("--thickness_thres", default=3.0)
    parser.add_argument("--size_hist_range", nargs='+',
        default=[0, 26300]
        )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data_paths = []
    evaluation = Evaluation(args.thickness_thres)
    if args.res_pkl_file == None or not os.path.exists(args.res_pkl_file):
        for folder in args.testset_dir:
            data_paths += get_infos(folder)
        probar = tqdm(total=len(data_paths))
        for im_file, anno_file, _ in data_paths:
            evaluation.single_patient_stats(anno_file)
            probar.update(1)
        probar.close()
    evaluation.set_stats(args.res_pkl_file, args.output_dir, args.size_hist_range)
