import json, os, pickle, sys
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, "../../..")
from fw_neutral.utils.data_proc import extra_processing
from fw_neutral.utils.config import Config
from fw_dependent.tf.model.UNet import UNet
from fw_dependent.tf.model.tf_layers import build_loss, average_gradients

class prc():
    def __init__(self, cfg):
        self.ex_process = extra_processing(cfg.im_size, cfg.num_class, cfg.preprocess, cfg.loss)

    def process(self, data_dir):
        data = pickle.load(open(data_dir, "rb"))
        return self.ex_process.preprocess(data["im"], data["mask"])

def create_ds(data_dirs, batch_size):
    ex_prc = prc(cfg)
    dataset = tf.data.Dataset.from_generator(lambda: (d for d in data_dirs), output_types=tf.string)
    dataset = dataset.map(lambda data_dir: tf.py_func(ex_prc.process, [data_dir], [tf.float32, tf.bool]), None)
    dataset = dataset.repeat().batch(batch_size).prefetch(512)
    ds_iter = dataset.make_one_shot_iterator()
    return ds_iter.get_next() # im: next_ele[0], gt: next_ele[1]

class tf_model():
    def __init__(self, cfg, num_gpus, training=True):
        data_dirs = json.load(open(cfg.trainset["data_list"], "r"))
        steps_per_epoch = len(data_dirs) // (num_gpus * cfg.batch_size) + 1
        in_im, in_gt = create_ds(data_dirs, cfg.batch_size)
        in_im, in_gt = tf.reshape(in_im, (cfg.batch_size, cfg.im_size[1], cfg.im_size[0], 1)), tf.reshape(in_im, (cfg.batch_size, cfg.im_size[1], cfg.im_size[0], 1))
        if training:
            global_step = tf.get_variable("global_step", 
                [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
            # self.global_step = global_step
            steps = [steps_per_epoch * epoch_step for epoch_step in cfg.lr_schedule["epoch_to_drop_lr"]]
            print(f"Steps to drop LR: {steps}")
            lr = tf.train.piecewise_constant(global_step, steps, cfg.lr_schedule["lr"])
            # self.learning_rate = lr
            optimizer = tf.train.AdamOptimizer(lr)

        if num_gpus > 1:
            tower_grads, loss_list = [], []
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                for i in range(num_gpus):
                    with tf.device(f"/gpu:{i}"):
                        with tf.name_scope(f"tower_{i}"):
                            pred = UNet(in_im, cfg)
                            if training:
                                loss = build_loss(pred, in_im, in_gt, cfg)
                                # loss_list.append(loss)
                                tower_grads.append(optimizer.compute_gradients(loss))
            
            if training:         
                mean_grad = average_gradients(tower_grads)
                # Aren't these 2 the same thing? (UPDATE_OPS-applying gradients)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    self.opt_op = optimizer.apply_gradients(mean_grad, global_step=global_step)
                # self.loss = tf.add_n(loss_list)
            
        elif num_gpus == 1:
            pred = UNet(in_im, cfg)
            if training: 
                self.opt_op = optimizer.minimize(build_loss(pred, in_im, in_gt, cfg), global_step=global_step)

if __name__ == "__main__":
    train_dir = "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0617/"
    cfg_file = "/rdfs/fast/home/sunyingge/pt_ground/configs/UNet_0615.json"
    cfg = Config()
    cfg.load_from_json(cfg_file)
    gpus = ["1", "3", "5"]
    # gpus = ["1"]
    num_gpus = len(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    sess = tf.Session()

    test_ds = False
    if test_ds:
        from fw_neutral.utils.data_proc import gen_data_list
        data_dirs, val_dirs = gen_data_list(train_dir, cfg)
        # data_dirs += val_dirs
        next_ele = create_ds(data_dirs, cfg.batch_size)
        # print(type(next_ele[0]))
        # next_ele[0].set_shape[(4, 256, 256, 1)]
        im = tf.reshape(next_ele[0], (8, 256, 256, 1))
        l = 10000
        pbar = tqdm(total=l)
        for _ in range(l):
            res = sess.run(im)
            # print(im[0].shape)
            print(res.shape)
            pbar.update(1)
        pbar.close()
    else:
        model = tf_model(cfg, num_gpus)
        l = 10000
        pbar = tqdm(total=l)
        sess.run(tf.global_variables_initializer())
        for _ in range(l):
            # res = sess.run(model.pred["seg_map"])
            sess.run(model.opt_op)
            pbar.update(1)
        pbar.close()