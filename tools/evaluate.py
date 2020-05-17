import argparse, os, sys

from collections import defaultdict
import cv2
import numpy as np
import pickle
import SimpleITK as sitk
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, "/rdfs/fast/home/sunyingge/code/chenfeng")
from misc.utils import get_infos, resize3d, graph_from_graphdef, pre_img_arr_1ch
from misc.my_metrics import dice_coef_pat
sys.path.insert(0, "/rdfs/fast/home/sunyingge/code/test/pt_ground/fast/")
from ASEUNet import SEResUNet

def show_dice(all_res):
    stats = defaultdict(list)
    for res in all_res:
        if 'covid_pneu_datasets' in res[0]:
            if res[2] >= args.thickness_thres:
                stats['covid'].append(res[1])
                stats['thick'].append(res[1])
                stats['covid_thick'].append(res[1])
            elif res[2] < args.thickness_thres:
                stats['covid'].append(res[1])
                stats['thin'].append(res[1])
                stats['covid_thin'].append(res[1])
        elif 'normal_pneu_datasets' in res[0]:
            if res[2] >= args.thickness_thres:
                stats['normal'].append(res[1])
                stats['thick'].append(res[1])
                stats['normal_thick'].append(res[1])
            elif res[2] < args.thickness_thres:
                stats['normal'].append(res[1])
                stats['thin'].append(res[1])
                stats['normal_thin'].append(res[1])
    for key in stats:
        print(f"{key}: {np.mean(np.array(stats[key]))}")

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("mode", choices=["eval", "calc_dice"],
        help="""
            'calc_dice' mode: 
                '--input_file': pickle file.
            'eval' mode: 
                '--input_file': Checkpoint file.
                "--testset_dir"
                "--pkl_dir"
            """
            )
    parser.add_argument("gpu")
    parser.add_argument("--input_file",
        # default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/epoch_9.ckpt",
        # default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/epoch_9_res.pkl"
        # default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0516/epoch_5.ckpt",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_22.ckpt",
        )
    parser.add_argument("--testset_dir", nargs='+',
        default=["/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/normal_pneu_datasets",
        "/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/covid_pneu_datasets"])
    parser.add_argument("--pkl_dir",
        # default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0516/epoch_5_res.pkl",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_22_res.pkl",
        )
    parser.add_argument("--thickness_thres", default=3.0)
    parser.add_argument("--batch_size", default=32)
    return parser.parse_args()

args = parse_args()

class Args(object):
    def __init__(self):
        self.img_size = (256, 256)

if args.mode == "eval":
    model_args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    info_paths = []
    for folder in args.testset_dir:
        info_paths += get_infos(folder)
    info_paths = sorted(info_paths, key=lambda info:info[0])
    all_result = []
    do_paths = info_paths

    input_im = tf.placeholder(tf.float32, shape=(None, model_args.img_size[0], model_args.img_size[1], 1))
    pred = SEResUNet(input_im, num_classes=1, reduction=8, name_scope="SEResUNet")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, args.input_file)
    # need_plot = True
    pbar = tqdm(total=len(do_paths))
    for num, info in enumerate(do_paths):
        img_file, lab_file = info[0:2]
        try:
            img_ori,  lab_ori  = sitk.ReadImage(img_file, sitk.sitkFloat32), sitk.ReadImage(lab_file, sitk.sitkInt16)
            img_arr,  lab_arr  = sitk.GetArrayFromImage(img_ori), sitk.GetArrayFromImage(lab_ori)
            lab_arr  = np.asarray(lab_arr > 0, dtype=np.uint8)
        except:
            continue
        depth, ori_shape = img_arr.shape[0], img_arr.shape[1:]
        # Thickness of the slice?
        spacing = img_ori.GetSpacing()
        # min_size = int(24/(np.prod(img_ori.GetSpacing())))
        
        dis_min_ch1 = -1400. ; dis_max_ch1 = 800. 
        dis_min_ch2 = -1000. ; dis_max_ch2 = -100.
        
    #     dis_arr = pre_img_arr_2ch(img_arr,
    #                              min_ch1=dis_min_ch1, max_ch1=dis_max_ch1,
    #                              min_ch2=dis_min_ch2, max_ch2=dis_max_ch2,
    #                              )

        dis_arr = pre_img_arr_1ch(img_arr, min_ch1=dis_min_ch1, max_ch1=dis_max_ch1)
        dis_arr = resize3d(dis_arr, (256, 256), interpolation=cv2.INTER_LINEAR)
        dis_arr = np.expand_dims(dis_arr, axis=-1)
        
        pred_ = []
        # segs =  16 # batch_size?
        segs = args.batch_size
        assert isinstance(segs, int) and (segs>0) & (segs<70), "Please" 
        step = depth//segs + 1 if depth%segs != 0 else depth//segs
        for ii in range(step):
            if ii != step-1:
                if False: print("stage1")
                if False: print(dis_arr[ii*segs:(ii+1)*segs, ...].shape)
                pp = sess.run(pred, feed_dict={input_im: dis_arr[ii*segs:(ii+1)*segs, ...]}) #[0]
                pp = 1/ (1 + np.exp(-pp))
                pred_.append(pp)
            else:
                if False: print("stage2")
                if False: print(dis_arr[ii*segs:, ...].shape)
                pp = sess.run(pred, feed_dict={input_im: dis_arr[ii*segs:, ...]}) #[0]
                pp = 1/ (1 + np.exp(-pp))
                pred_.append(pp)
        dis_prd = np.concatenate(pred_, axis=0)
        dis_prd = (dis_prd > 0.5)
        dis_prd = resize3d(dis_prd.astype(np.uint8), ori_shape, interpolation=cv2.INTER_NEAREST)
        score = dice_coef_pat(dis_prd, lab_arr)
        if score < 0.3:
            print(os.path.dirname(lab_file))
            print(score)
        all_result.append([img_file, score, round(spacing[-1], 1)])
        # if False:
        #     pos = (lab_arr>0.5).any(axis=(-1, -2))
        #     pos_img = img_arr[pos]; pos_prd = dis_prd[pos]; pos_lab = lab_arr[pos]
        #     pos_depth = pos_img.shape[0]
        #     max_depth = min(pos_depth, 20)
        #     for ii in range(0, max_depth, 2):
        #         plt.subplot(131)
        #         plt.imshow(pos_img[ii, ...], cmap="gray")
        #         plt.subplot(132)
        #         plt.imshow(pos_img[ii, ...], cmap="gray")
        #         plt.imshow(pos_prd[ii, ...], alpha=0.5)
        #         plt.subplot(133)
        #         plt.imshow(pos_img[ii, ...], cmap="gray")
        #         plt.imshow(pos_lab[ii, ...], alpha=0.5)
        #         plt.show()

        # if False:
        #     ori_dir = os.path.dirname(img_file)
        #     dir_path = ori_dir.replace("", "")
        #     #assert dir_path is not ori_dir, "check"
        #     if not os.path.exists(dir_path): os.makedirs(dir_path)
        #     basename = os.path.basename(img_file)
        #     pred_dis_ori = sitk.GetImageFromArray(dis_prd)
        #     pred_dis_ori.CopyInformation(img_ori)
        #     #shutil.copyfile(img_file, os.path.join(dir_path, basename))
        #     #sitk.WriteImage(img_ori, os.path.join(dir_path, basename))
        #     save_file = os.path.join(dir_path, basename.replace(".nii.gz", "_prd-label.nii.gz"))
        #     assert save_file != img_file
        #     sitk.WriteImage(pred_dis_ori, save_file)
        
        pbar.update(1)
    pbar.close()
    if os.path.exists(args.pkl_dir):
        input("Result file already exists. Press enter to continue and overwrite it...")
    pickle.dump(all_result, open(args.pkl_dir, "bw"))
    show_dice(all_result)
elif args.mode == "calc_dice":
    with open(args.input_file, "br") as f:
        all_res = pickle.load(f)
    show_dice(all_res)
