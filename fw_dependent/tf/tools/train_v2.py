import argparse, json, os, pickle, sys
import logging
from os.path import join as pj
# from pprint import pprint
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, "../../..")
from fw_neutral.utils.data_proc import extra_processing
from fw_neutral.utils.dataset import finalize_data_dirs
from fw_neutral.utils.traineval import gen_outdirs
from fw_neutral.utils.config import Config
from fw_dependent.tf.model.UNet import UNet
from fw_dependent.tf.model.tf_layers import average_gradients, build_loss, choose_model

def parse_args():
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("mode", choices=["train", "val", "speed_test"])
    parser.add_argument("config")
    parser.add_argument("-o", "--output_dir",
        help="If a full existing dir is provided, an auto created one won't be used.",
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_0611/UNet_")
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_0611/Test_0628")
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0611/UNet_0628")
    parser.add_argument("--gpus", nargs="+")
    parser.add_argument("--min_num_workers", help="Set to #gpus if this is smaller", default=2)
    
    parser.add_argument("--train_dir", help="Training set directory.",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0617/")
    parser.add_argument("--resume")
    parser.add_argument("--train_debug", action="store_true")
    parser.add_argument("--eval_in_train", action="store_true", default=True)
    
    parser.add_argument("--testset_dir", nargs="+",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Test_0616")

    parser.add_argument("--loss_logging_fq", type=int, default=5)

    return parser.parse_args()

def training_create_ds(args, cfg, num_threads, result_dir):
    data_dirs, val_dirs = finalize_data_dirs(args.mode, result_dir, args.train_dir, args.testset_dir, cfg)
    print(f"Num of training samples: {len(data_dirs)}")
    print(f"Num of validation samples: {len(val_dirs)}")
    output_types = (tf.float32, tf.float32, tf.int64, tf.string, tf.float32)
    output_shapes = (tf.TensorShape((None, cfg.im_size[1], cfg.im_size[0], 1)), 
        tf.TensorShape((None, cfg.im_size[1], cfg.im_size[0], 1)),
        tf.TensorShape((None, 2)), tf.TensorShape((None)),
        # Since we are not performing any tf ops on og ims so their size info shouldn't matter.
        # This same character works for others as well as long as we don't care about their size info.
        tf.TensorShape((None))) 
    debug = False
    iterator = tf.data.Iterator.from_structure(output_types, output_shapes) #
    ex_prc = extra_processing(cfg.im_size, cfg.num_class, cfg.preprocess, cfg.loss)

    train_steps_per_epoch = len(data_dirs) // (num_gpus * cfg.batch_size) + 1
    train_ds = tf.data.Dataset.from_generator(lambda: (d for d in data_dirs), output_types=tf.string)
    train_ds = train_ds.map(lambda data_dir: tf.py_func(ex_prc.train_process_tfdata, [data_dir], output_types), num_threads)
    train_ds = train_ds.repeat().batch(cfg.batch_size).prefetch(512)
    train_init = iterator.make_initializer(train_ds) #

    val_num_batch = len(val_dirs) // (num_gpus * cfg.batch_size) + 1
    val_ds = tf.data.Dataset.from_generator(lambda: (d for d in val_dirs), output_types=tf.string)
    val_ds = val_ds.map(lambda data_dir: tf.py_func(ex_prc.val_process_tfdata, [data_dir], output_types), num_threads)
    val_ds = val_ds.batch(cfg.batch_size).prefetch(512)
    val_init = iterator.make_initializer(val_ds) #

    ds_handles = dict(next_ele=iterator.get_next(), 
        train_init=train_init,
        val_init=val_init,
        train_steps_per_epoch=train_steps_per_epoch,
        val_num_batch=val_num_batch)
    return ds_handles

class tf_model():
    def __init__(self, cfg, num_gpus, min_num_workers, result_dir, training=True):
        network_class = choose_model(cfg.network["name"])
        if training:
            ds_handles = training_create_ds(args, cfg, max(num_gpus, min_num_workers), result_dir)
            in_im = ds_handles["next_ele"][0]
            in_gt = ds_handles["next_ele"][1]
            self.in_im, self.in_gt = in_im, in_gt
            self.tl = ds_handles["next_ele"][2]
            self.in_dir = ds_handles["next_ele"][3]
            self.og_in_im = ds_handles["next_ele"][4]
            self.train_init, self.val_init = ds_handles["train_init"], ds_handles["val_init"]
            self.train_steps_per_epoch, self.val_num_batch = ds_handles["train_steps_per_epoch"], ds_handles["val_num_batch"]
            global_step = tf.get_variable("global_step", [], trainable=False, dtype=tf.int32, 
                initializer=tf.constant_initializer(-cfg.warmup_steps if cfg.warmup else 0))
            if cfg.lr_schedule["type"] == "epoch_wise_constant":
                learning_rate = tf.get_variable("learning_rate", 
                    initializer=cfg.lr_schedule["lr"][0], trainable=False)
                steps = [self.train_steps_per_epoch * epoch_step for epoch_step in cfg.lr_schedule["epoch_to_drop_lr"]]
                print(f"Steps to drop LR: {steps}")
                lr = tf.train.piecewise_constant(global_step, steps, cfg.lr_schedule["lr"])
                init_lr = cfg.lr_schedule["lr"][0]
            elif cfg.lr_schedule["type"] == "halved":
                raise Exception("Not supported yet")
                learning_rate = tf.get_variable("learning_rate", 
                    initializer=cfg.lr_schedule["init_lr"], trainable=False)
                init_lr = cfg.lr_schedule["init_lr"]
            elif cfg.lr_schedule["type"] == "cos_decay":
                learning_rate = tf.train.cosine_decay(cfg.lr_schedule["lr"], global_step, 
                    cfg.lr_schedule["epoch_period"] * self.train_steps_per_epoch)
                init_lr = cfg.lr_schedule["lr"]
            if cfg.warmup:
                warmup_init_lr = init_lr * cfg.warmup_init_multiplier
                wu_lr = warmup_init_lr + (cfg.warmup_steps - tf.abs(global_step)) / cfg.warmup_steps * (init_lr - warmup_init_lr)
                lr_coeff = tf.cast(global_step >= 0, tf.float32)
                learning_rate = learning_rate.assign(lr_coeff * learning_rate + (1 - lr_coeff) * tf.cast(wu_lr, tf.float32))
            # tf.summary.scalar("LR", learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)

        if num_gpus > 1:
            tower_grads, loss_list = [], []
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                for i in range(num_gpus):
                    with tf.device(f"/gpu:{i}"):
                        with tf.name_scope(f"tower_{i}"):
                            pred = network_class(in_im, cfg)
                            if training:
                                loss = build_loss(pred, in_im, in_gt, cfg)
                                loss_list.append(loss)
                                tower_grads.append(optimizer.compute_gradients(loss))
            
            if training:         
                mean_grad = average_gradients(tower_grads)
                # Aren't these 2 the same thing? (UPDATE_OPS-applying gradients)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    self.opt_op = [optimizer.apply_gradients(mean_grad, global_step=global_step)]
                loss = tf.add_n(loss_list) / num_gpus            
        elif num_gpus == 1:
            pred = network_class(in_im, cfg)
            loss = build_loss(pred, in_im, in_gt, cfg)
            if training: 
                self.opt_op = [optimizer.minimize(loss, global_step=global_step)]
        if training:
            # tf.summary.scalar("Loss", loss)
            # self.sum_op = tf.summary.merge_all()
            self.loss = loss
            self.lr = learning_rate
            self.global_step = global_step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logging.info(update_ops)
        # if cfg.network["norm_layer"] == "BN_layers":
        #     self.opt_op += update_ops
        self.opt_op += update_ops

def train(sess, cfg, args, num_gpus):
    outdirs = gen_outdirs(args, "tf_data", args.config, args.train_debug)
    output_dir, out_res_dir = outdirs["output_dir"], outdirs["out_res_dir"]
    logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=pj(out_res_dir, "training.log"),
        filemode="a")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console)
    
    model = tf_model(cfg, num_gpus, args.min_num_workers, out_res_dir)
    saver = tf.train.Saver(max_to_keep=cfg.max_epoch)
    if args.resume:
        saver.restore(args.resume)
    else:
        sess.run(tf.global_variables_initializer())
        epoch = 1
    while epoch < cfg.max_epoch + 1:
        pbar = tqdm(total=model.train_steps_per_epoch)
        logging.debug(f"\nStarting epoch {epoch}.\n")
        sess.run(model.train_init)
        loss_sum = 0
        for idx in range(10 if args.train_debug else model.train_steps_per_epoch):
            res = sess.run([model.loss, model.lr, model.global_step] + model.opt_op)
            loss_ret, lr_ret = res[0], res[1]
            loss_sum += loss_ret
            if (idx + 1) % args.loss_logging_fq == 0:
                if args.train_debug:
                    logging.info(f"Loss: {loss_sum / args.loss_logging_fq}, LR: {lr_ret}")
                else:
                    logging.debug(f"Loss: {loss_sum / args.loss_logging_fq}, LR: {lr_ret}")
                loss_sum = 0
            pbar.update(1)
        pbar.close()
        ckpt_dir = pj(output_dir, f"epoch_{epoch}.ckpt")
        saver.save(sess, ckpt_dir)
        epoch += 1

        if args.eval_in_train:
            pbar = tqdm(total=model.val_num_batch)
            sess.run(model.val_init)
            for _ in range(2000 if args.train_debug else model.val_num_batch):
                res = sess.run([model.og_in_im])
                # print(res[0].shape)
                pbar.update(1)
            pbar.close()

def evaluation(sess, model):
    pass

if __name__ == "__main__":
    args = parse_args()
    if args.mode != "speed_test":
        cfg = Config()
        cfg.load_from_json(args.config)
        num_gpus = len(args.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)
        sess = tf.Session()
        if args.mode == "train":
            train(sess, cfg, args, num_gpus)
        elif args.mode =="val":
            model = tf_model(cfg, num_gpus, args.min_num_workers, False)
            evaluation(sess, None)
    else:
        choices = ["test_ds", "test_train", "test_val"]
        test_mode = choices[1]
        l = 10000
        train_dir = "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0617/"
        # cfg_file = "/rdfs/fast/home/sunyingge/pt_ground/configs/UNet_0615.json"
        cfg_file = "/rdfs/fast/home/sunyingge/data/models/workdir_0611/UNet_fpfn_0624_01/config.json"
        gpus = ["6", "7"]

        cfg = Config()
        cfg.load_from_json(cfg_file)
        num_gpus = len(gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        sess = tf.Session()
        if test_mode == "test_ds":
            from fw_neutral.utils.data_proc import gen_data_list
            data_dirs, val_dirs = gen_data_list(train_dir, cfg)
            # data_dirs += val_dirs
            next_ele = create_ds(cfg, data_dirs)
            # print(type(next_ele[0]))
            # next_ele[0].set_shape[(4, 256, 256, 1)]
            im = tf.reshape(next_ele[0], (8, 256, 256, 1))
            pbar = tqdm(total=l)
            for _ in range(l):
                res = sess.run(im)
                # print(im[0].shape)
                print(res.shape)
                pbar.update(1)
            pbar.close()
        elif test_mode == "test_train":
            model = tf_model(cfg, num_gpus, args.min_num_workers)
            pbar = tqdm(total=l)
            sess.run(tf.global_variables_initializer())
            for _ in range(l):
                # res = sess.run(model.pred["seg_map"])
                sess.run(model.opt_op)
                sess.run(eval_model.opt_op)
                pbar.update(1)
            pbar.close()