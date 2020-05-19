import argparse, os, sys, time
import cv2, pickle
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from collections import defaultdict
from os.path import join as pj
from tqdm import tqdm

# When the TF module is large enough create a seperate folder for it
# TODO: Change working directory to point to folder of this script
sys.path.insert(0, "..") 
# from fast.cf_mod.misc.data_tools import BaseDataset, paths_for_dataset
from fast.cf_mod.misc.data_tools import BaseDataset
from utils.data_proc import paths_from_data
# TODO: unify the process of building models
from fast.ASEUNet import SEResUNet

# Older version eval import
sys.path.insert(0, "/rdfs/fast/home/sunyingge/code/chenfeng")
from misc.utils import get_infos, resize3d, graph_from_graphdef, pre_img_arr_1ch
from misc.my_metrics import dice_coef_pat

def parse_args():
    parser = argparse.ArgumentParser(
        """
        You need to be careful with resuming while also changing the batch size 
        with this script since it would change how many steps to take to drop LR.
        
        "eval" mode: 
            "--model_file": Checkpoint file.
            "--testset_dir"
            "--pkl_dir": Output.
        """)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("gpu", help="Choose GPU.")
    parser.add_argument("num_cls", type=int)
    
    # TODO: add an automatically generated (time-related) postfix
    parser.add_argument("-o", "--output_dir",
        help="""You only need to provide a prefix which will be automatically be 
            complemented by time to keep things distincive easily.""",
        default="/rdfs/fast/home/sunyingge/data/models/workdir_multicat_0518/SEResUNET_")

    # Training mode related
    parser.add_argument("--train_dir", help="Training set directory.",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0518/",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0519/",
        )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume", help="Checkpoint file to resume from.",
        # default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_11.ckpt")
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_multicat_0518/SEResUNET_0518_1902/epoch_1.ckpt"
    )

    # Eval mode related
    parser.add_argument("--model_file",
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_multicat_0518/SEResUNET_0518_1902/epoch_2.ckpt"
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_22.ckpt",
        )
    parser.add_argument("--testset_dir", nargs='+',
        default=["/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/normal_pneu_datasets",
        "/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/covid_pneu_datasets"]
        )
    parser.add_argument("--pkl_dir",
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_multicat_0518/SEResUNET_0518_1902/epoch_2_res.pkl",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_22_res.pkl",
        )
    parser.add_argument("--thickness_thres", default=3.0)

    return parser.parse_args()
    # So that this works with jupyter
    # return parser.parse_args(args=[
    #     "--train_dir",
    #     "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train/",
    #     "--output_dir",
    #     "/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/"
    # ])

class Args(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.img_size = (256, 256)
        self.epochs = 40
        # The network would behave quite differently when n_classes > 1
        # and when n_classes == 1. The actual number of classes will be this 
        # plus 1 (BG).
        self.n_classes = args.num_cls
        # self.n_channels = 2
        # self.initial_epoch = 0
        # self.global_step = 0
        # self.lr = 0.0005
        # self.percentage = 0.1
        # self.load = False
        # self.save_cp = True
        # self.description = "HRNET_KERAS_DEFAULT"

        self.steps = [4, 8] # The unit is epoch
        self.lr = [1e-3, 5e-4, 25e-5]
        self.max_to_keep = 30

class tf_model():
    def __init__(self, args, training_args, num_batches=None):
        self.input_im = tf.placeholder(tf.float32, shape=(None, training_args.img_size[0], training_args.img_size[1], 1))
        if training_args.n_classes == 1:
            self.pred = SEResUNet(self.input_im, num_classes=1, reduction=8, name_scope="SEResUNet")
        else:
            self.pred = SEResUNet(self.input_im, num_classes=training_args.n_classes + 1, reduction=8, name_scope="SEResUNet")
        
        if args.mode == "train":
            if training_args.n_classes == 1:
                # For some reason float32 is working
                self.input_ann = tf.placeholder(tf.float32, shape=(None, training_args.img_size[0], training_args.img_size[1], 1))
                self.ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_ann, logits=self.pred)
            else:
                self.input_ann = tf.placeholder(tf.int32, shape=(None, training_args.img_size[0], training_args.img_size[1]))
                self.ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_ann, logits=self.pred)
            self.loss = tf.reduce_mean(self.ce)
            global_step = tf.Variable(0, trainable=False)
            assert num_batches, "Please provide batch size for training mode!"
            steps = [num_batches * epoch_step for epoch_step in training_args.steps]
            print(f"Steps to drop LR: {steps}")
            learning_rate = tf.train.piecewise_constant(global_step, steps, training_args.lr)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

def ini_training_set(args, training_args):
    print("==>>Training set: ")
    # _, train_pos, _ = paths_for_dataset(args.train_dir,
    #     flags=["train"],
    #     seed=999,
    #     isprint=True)
    train_paths  = paths_from_data(args.train_dir)
    np.random.seed(999)
    train_paths = np.random.permutation(train_paths).tolist()
    print("++"*30)
    print(f"Number of training samples: {len(train_paths)}")
    train_dataset = BaseDataset(train_paths, [], img_size=training_args.img_size, choice="all",
        image_key="im", mask_key="mask")
    print(f"train_dataset: {len(train_dataset)}")
    return train_dataset

def train(sess, args, training_args):
    train_dataset = ini_training_set(args, training_args)
    num_batches = len(train_dataset) // training_args.batch_size
    model = tf_model(args, training_args, num_batches)
    if args.resume:
        output_dir = os.path.dirname(args.resume)
    else:
        output_dir = args.output_dir + time.strftime("%m%d_%H%M", time.localtime())
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    saver = tf.train.Saver(max_to_keep=training_args.max_to_keep)
    num_para = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("Total number of trainable parameters: {:.3}M.\n".format(float(num_para)/1e6))
    if args.resume:
        saver.restore(sess, args.resume)
        # This needs to be changed if the naming rule changes
        epoch = int((os.path.basename(args.resume).split('.')[0]).split('_')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        epoch = 0
    while epoch < training_args.epochs:
        print(f"Epoch {epoch + 1}\n")
        for i in range(num_batches):
        # for i in range(10):
            print(f"Epoch progress: {i + 1} / {num_batches}")
            data_list = []
            for j in range(i * training_args.batch_size, (i + 1) * training_args.batch_size):
                data_list.append(train_dataset[j])
            data_ar = np.array(data_list)
            im_ar, ann_ar = data_ar[:,0,:,:,:], data_ar[:,1,:,:,:] if training_args.n_classes == 1 else data_ar[:,1,:,:,0]

            ret_loss, ret_pred, ret_ce, _ = sess.run([model.loss, model.pred, model.ce, model.optimizer],
                feed_dict={model.input_im: im_ar, model.input_ann: ann_ar,})
            print(ret_loss)
        saver.save(sess, pj(output_dir, f"epoch_{epoch + 1}.ckpt"))
        epoch += 1

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

def evaluation(sess, args, training_args):
    info_paths = []
    for folder in args.testset_dir:
        info_paths += get_infos(folder)
    info_paths = sorted(info_paths, key=lambda info:info[0])
    all_result = []
    do_paths = info_paths
    model = tf_model(args, training_args)
    saver = tf.train.Saver()
    saver.restore(sess, args.model_file)
    pbar = tqdm(total=len(do_paths))
    if os.path.exists(args.pkl_dir):
        input("Result file already exists. Press enter to \
            continue and overwrite it when inference is done...")
    for num, info in enumerate(do_paths):
        img_file, lab_file = info[0:2]
        try:
            img_ori,  lab_ori  = sitk.ReadImage(img_file, sitk.sitkFloat32), sitk.ReadImage(lab_file, sitk.sitkInt16)
            img_arr,  lab_arr  = sitk.GetArrayFromImage(img_ori), sitk.GetArrayFromImage(lab_ori)
            lab_arr  = np.asarray(lab_arr > 0, dtype=np.uint8)
        except:
            continue
        depth, ori_shape = img_arr.shape[0], img_arr.shape[1:]
        spacing = img_ori.GetSpacing()
        dis_min_ch1 = -1400. ; dis_max_ch1 = 800. 
        dis_arr = pre_img_arr_1ch(img_arr, min_ch1=dis_min_ch1, max_ch1=dis_max_ch1)
        dis_arr = resize3d(dis_arr, (256, 256), interpolation=cv2.INTER_LINEAR)
        dis_arr = np.expand_dims(dis_arr, axis=-1)
        
        pred_ = []
        segs = training_args.batch_size
        assert isinstance(segs, int) and (segs>0) & (segs<70), "Please" 
        step = depth//segs + 1 if depth%segs != 0 else depth//segs
        for ii in range(step):
            if ii != step-1:
                if False: print("stage1")
                if False: print(dis_arr[ii*segs:(ii+1)*segs, ...].shape)
                pp = sess.run(model.pred, feed_dict={model.input_im: dis_arr[ii*segs:(ii+1)*segs, ...]}) #[0]
                pp = 1/ (1 + np.exp(-pp))
                pred_.append(pp)
            else:
                if False: print("stage2")
                if False: print(dis_arr[ii*segs:, ...].shape)
                pp = sess.run(model.pred, feed_dict={model.input_im: dis_arr[ii*segs:, ...]}) #[0]
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
        pbar.update(1)
    pbar.close()
    pickle.dump(all_result, open(args.pkl_dir, "bw"))
    show_dice(all_result)

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    training_args = Args(args)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)

    if args.mode == "train":
        train(sess, args, training_args)
    elif args.mode == "eval":
        evaluation(sess, args, training_args)