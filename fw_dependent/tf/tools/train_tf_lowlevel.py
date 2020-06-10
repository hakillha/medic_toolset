import argparse, logging, os, sys, time
import pickle, random, shutil
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from collections import defaultdict
from os.path import join as pj
from tqdm import tqdm

# When the TF module is large enough create a seperate folder for it
# TODO: Change working directory to point to folder of this script
sys.path.insert(0, "../../..") 
# from fast.cf_mod.misc.data_tools import BaseDataset, paths_for_dataset
from fast.cf_mod.misc.data_tools import BaseDataset
from fast.cf_mod.misc.utils import get_infos
from fast.cf_mod.misc.my_metrics import dice_coef_pat
# TODO: unify the process of building models
from fw_dependent.tf.model.tf_layers import tf_model
from fw_neutral.utils.config import Config
from fw_neutral.utils.data_proc import extra_processing, im_normalize, paths_from_data, Pneu_type
from fw_neutral.utils.viz import viz_patient
from fw_neutral.utils.metrics import Evaluation, show_dice

def parse_args():
    parser = argparse.ArgumentParser(
        """
        You need to be careful with resuming while also changing the batch size 
        with this script since it would change how many steps to take to drop LR.
        
        "eval" mode mostly requires: 
            "--model_file": Checkpoint file.
            
            "--testset_dir"
            "--pkl_dir": Output. Optional now.

        "eval_multi" mode mostly requires:
            "--mode_file"
            "--pkl_dir": This is pretty much deprecated.
            "--epochs2eval"
        """)
    parser.add_argument("mode", choices=["train", "eval", "eval_multi"])
    parser.add_argument("config", help="Config file.")
    parser.add_argument("--gpu", help="Choose GPU.")
    parser.add_argument("--gpus_to_use", nargs='+')
    
    # Training mode related
    parser.add_argument("-o", "--output_dir",
        help="""You only need to provide a prefix which will be automatically be 
            complemented by time to keep things distincive easily. This will be ignored
            when resuming from a checkpoint.
            You can also provide one that already exists to overwrite the automatical generated one.""",
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_")
    parser.add_argument("--train_dir", help="Training set directory.",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0519/",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0526/",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0608/",
        )
    parser.add_argument("--batch_size", type=int,
        help="Provided here to enable easy overwritting (particularly useful for evaluation).")
    parser.add_argument("--resume", 
        help="""Checkpoint file to resume from. Its directory will overwrite "output_dir".""",
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNet_0526_01/epoch_3.ckpt"
        )
    parser.add_argument("--max_to_keep", default=30, help="Max number of checkpoint files to keep.")
    parser.add_argument("--num_retry", default=60)
    parser.add_argument("--retry_waittime", default=120, help="In seconds.")
    parser.add_argument("--eval_while_train", action="store_true", default=True,
        help="""Need to provide "--testset_dir" for this.""")
    parser.add_argument("--debug", action="store_true", 
        help="""When set to true, run only 10 steps 
        to see if eval after checkpoint works correctly.""")

    # Eval mode related
    parser.add_argument("--testset_dir", nargs='+',
        default=["/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/0519/normal_pneu_datasets",
        "/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/0519/covid_pneu_datasets"]
        )
    parser.add_argument("--model_file",
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNet_0601_02/epoch_9.ckpt",
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0602_1348_05/epoch_1.ckpt"
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0528_1357_35/epoch_12.ckpt"
        )
    parser.add_argument("--pkl_dir",
        help="""If not provided, it will be generated from "model_file".""",
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNet_0601_02/epoch_9_res.pkl",
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0602_1348_05/epoch_1_res.pkl"
        )
    parser.add_argument("--epochs2eval", nargs='+', default=["6", "5"])
    parser.add_argument("--thickness_thres", default=3.0)
    parser.add_argument("--viz", help="Middle name of the visualization output directory.")
    parser.add_argument("--eval_debug", action="store_true")

    return parser.parse_args()
    # So that this works with jupyter
    # return parser.parse_args(args=[
    #     "eval",
    #     "/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0528_1357_35/cfg.json",
    #     "--gpu",
    #     "2",
    #     "--batch_size",
    #     "16"
    #     "--viz",
    #     "viz_0530_01"
    # ])

def ini_training_set(args, cfg):
    print("==>>Training set: ")
    print("++"*30)
    train_dataset = BaseDataset(paths_from_data(args.train_dir, "pos"), 
        paths_from_data(args.train_dir, "neg"),
        img_size=cfg.im_size, choice="all", image_key="im", mask_key="mask")
    print(f"Number of training samples: {len(train_dataset)}")
    return train_dataset

def train(sess, args, cfg):
    train_dataset = ini_training_set(args, cfg)
    num_batches = len(train_dataset) // cfg.batch_size
    model = tf_model(args, cfg, args.gpus_to_use, num_batches)
    if args.resume:
        output_dir = os.path.dirname(args.resume)
    else:
        if os.path.exists(args.output_dir):
            output_dir = args.output_dir
        else:
            # If out_dir is not overwritten by an existing one,
            # create one automatically.
            output_dir = args.output_dir + time.strftime("%m%d_%H%M_%S", time.localtime())
        if os.path.exists(output_dir):
            if not args.debug:
                input("The output directory already exists, please wait a moment and restart...")
                # print("The output directory already exists, please wait a moment and restart...")
                # sys.exit()
        else:
            os.makedirs(output_dir)
        if not os.path.exists(pj(output_dir, os.path.basename(args.config))):
            shutil.copy(args.config, output_dir)
    logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=pj(output_dir, "training.log"),
        filemode="a")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console)
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)
    num_para = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    logging.info("Total number of trainable parameters: {:.3}M.\n".format(float(num_para)/1e6))
    if args.resume:
        # So that an extended model can use pretrained weights?
        # Add another args for this? Does this affect the (free) model checking otherwise?
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.resume)
        # This needs to be changed if the naming rule changes
        epoch = int((os.path.basename(args.resume).split('.')[0]).split('_')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        epoch = 0
    while epoch < cfg.max_epoch:
        logging.info(f"Epoch {epoch + 1}\n")
        num_batches = 10 if args.debug else num_batches
        # with tf.contrib.tfprof.ProfileContext("", trace_steps=[], dump_steps=[]) as pctx:
        for i in range(num_batches):
            data_list = []
            for j in range(i * cfg.batch_size, (i + 1) * cfg.batch_size):
                # Careful with a batch size that can't be divided evenly by the number of gpus
                # when at the end during a multi-GPU training
                ex_process = extra_processing(cfg)
                im_ar, ann_ar = ex_process.preprocess(train_dataset[j][0], train_dataset[j][1], True)
                data_list.append((im_ar, ann_ar))
            data_ar = np.array(data_list)
            ret_loss, ret_lr, _, global_step = sess.run(
                [model.loss, model.learning_rate, model.opt_op, model.global_step],
                feed_dict={model.input_dict["im"]: data_ar[:,0,:,:,:], 
                model.input_dict["anno"]: data_ar[:,1,:,:,:],})
            # if i % 5 == 0:
            logging.info(f"Epoch progress: {i + 1} / {num_batches}, loss: {ret_loss:.5f}, lr: {ret_lr:.8f}, global step: {global_step}")
        for _ in range(args.num_retry):
            try:
                ckpt_dir = pj(output_dir, f"epoch_{epoch + 1}.ckpt")
                saver.save(sess, ckpt_dir)
                # a = np.random.uniform(size=1)#
                # if a[0] < 0.9:#
                #     raise Exception("Hi!")#
                break
            except:
                logging.warning("Failed to save checkpoint. Retry after 2 minutes...")
                time.sleep(args.retry_waittime)
                # time.sleep(10)#
        if args.eval_while_train and not args.gpus_to_use:
            evaluation("during_training", sess, args, cfg, model, ckpt_dir.replace(".ckpt", "_res.pkl"), log=True)
        epoch += 1

def evaluation(mode, sess, args, cfg, model=None, pkl_dir=None, log=False):
    """
        Args:
            mode: ["during_training", "eval", "eval_mutli"]
            model: For eval during training.
            pkl_dir: Provided for an eval during training to overwrite the args.
    """
    eval_object = Evaluation(args.thickness_thres)
    info_paths = []
    for folder in args.testset_dir:
        info_paths += get_infos(folder)
    if args.eval_debug:
        random.shuffle(info_paths)
    else:
        info_paths = sorted(info_paths, key=lambda info:info[0])
    all_result = []

    if mode == "eval": # This mode is deprecated
        model = tf_model(args, cfg)
        saver = tf.train.Saver()
        saver.restore(sess, args.model_file)
    elif mode == "during_training":
        args.pkl_dir = pkl_dir
    elif mode == "eval_multi":
        saver = tf.train.Saver()
        saver.restore(sess, args.model_file)

    pbar = tqdm(total=len(info_paths))
    if os.path.exists(args.pkl_dir):
        input("Result file already exists. Press enter to \
            continue and overwrite it when inference is done...")
    for info in info_paths:
        img_file, lab_file = info[0:2]
        try:
            img_ori,  lab_ori  = sitk.ReadImage(img_file, sitk.sitkFloat32), sitk.ReadImage(lab_file, sitk.sitkInt16)
            img_arr,  lab_arr  = sitk.GetArrayFromImage(img_ori), sitk.GetArrayFromImage(lab_ori)
        except:
            continue
        depth, ori_shape = img_arr.shape[0], img_arr.shape[1:]
        spacing = img_ori.GetSpacing()
        img_arr_normed = im_normalize(img_arr, cfg.preprocess["normalize"]["ct_interval"], 
            cfg.preprocess["normalize"]["norm_by_interval"])
        ex_processing = extra_processing(cfg, og_shape=ori_shape[::-1])
        dis_arr, lab_arr = ex_processing.batch_preprocess(img_arr_normed, lab_arr, False)

        pred_ = []
        segs = cfg.batch_size
        assert isinstance(segs, int) and (segs>0) & (segs<70), "Please" 
        step = depth//segs + 1 if depth%segs != 0 else depth//segs
        for ii in range(step):
            if ii != step-1:
                pp = sess.run(model.pred["seg_map"], feed_dict={model.input_dict["im"]: dis_arr[ii*segs:(ii+1)*segs, ...]}) #[0]
            else:
                pp = sess.run(model.pred["seg_map"], feed_dict={model.input_dict["im"]: dis_arr[ii*segs:, ...]}) #[0]
            pp = 1/ (1 + np.exp(-pp)) # this only works for single class
            pred_.append(pp)
        dis_prd = np.concatenate(pred_, axis=0)
        # add the og version in
        if cfg.num_class == 1:
            dis_prd = dis_prd > 0.5
        else:
            pneumonia_type = Pneu_type(img_file, False)
            if pneumonia_type == "common_pneu":
                cls_id = 2
            elif pneumonia_type == "covid_pneu":
                cls_id = 1
            else:
                raise Exception("Unknown condition!")
            dis_prd = np.argmax(dis_prd, -1) == cls_id
        dis_prd = ex_processing.batch_postprocess(dis_prd)
        if args.eval_debug:
            pred_nii = sitk.GetImageFromArray(dis_prd)
            pred_nii.CopyInformation(lab_ori)
            im_dir_list = img_file.split('/')
            debug_out = pj(os.path.dirname(args.model_file), "eval_debug", im_dir_list[-3], im_dir_list[-2])
            if not os.path.exists(debug_out):
                os.makedirs(debug_out)
            sitk.WriteImage(pred_nii, 
                pj(debug_out, os.path.basename(img_file).replace(".nii.gz", "_pred.nii.gz")))
            shutil.copy(img_file, debug_out)
            shutil.copy(lab_file, debug_out)
        else:
            # eval_object.eval_single_patient(lab_file, dis_prd)
            score = dice_coef_pat(dis_prd, lab_arr)
            if score < 0.3:
                if args.viz:
                    viz_patient(img_arr_normed, dis_prd, lab_arr, 
                        pj(os.path.dirname(args.model_file), args.viz), img_file)
                if log:
                    logging.info(os.path.dirname(lab_file))
                    logging.info(score)
                else:
                    print(os.path.dirname(lab_file))
                    print(score)
            all_result.append([img_file, score, round(spacing[-1], 1)])
        pbar.update(1)
    pbar.close()
    pickle.dump(all_result, open(args.pkl_dir, "bw"))
    show_dice(all_result, log=log)

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.gpu else ",".join(args.gpus_to_use)
    cfg = Config()
    cfg.load_from_json(args.config)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if not args.pkl_dir:
        args.pkl_dir = args.model_file.replace(".ckpt", "_res.pkl")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)

    if args.mode == "train":
        train(sess, args, cfg)
    elif args.mode == "eval":
        evaluation("eval", sess, args, cfg)
    elif args.mode == "eval_multi":
        model = tf_model(args, cfg)
        for epoch in args.epochs2eval:
            args.model_file = pj(os.path.dirname(args.model_file), f"epoch_{epoch}.ckpt")
            args.pkl_dir = pj(os.path.dirname(args.pkl_dir), f"epoch_{epoch}_res.pkl")
            evaluation("eval_multi", sess, args, cfg, model)
            print(f"Finished evaluating epoch {epoch}.")