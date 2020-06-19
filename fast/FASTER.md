# Changelog

# Task
## Multiclass segmentation
2020/05/18: Covid pneu is labeled with 1 and common pneu is labeled with 2 now.

# Training
> Training procedures:  
> 1. `fast/cf_mod/demon.py` for data preparation.
> 2. `tools/train_tf_lowlevel.py`
> 3. `tools/evaluate.py`
## `train_tf_lowlevel.py`
TODO: Add a specific name to ease the lookup for the input/output tensors.

## Training meta-parameters
2020/05/18: The parameters that you should care about:
> Clipping thresholds; reduction for SEResUNet(=8 currently); min_ct_1ch=-1400, max_ct_1ch=800; optimizer: Adam with LR=[1e-3, 5e-4, 25e-5] decreasing at epoch [4, 8].

# TODO
* Save tensorboard log
* Keep a separate text file to keep a good track of all the files in each split set and to keep everyone's data consistent.
* Check files integrity and shape consistency between images and annoataions.

# Useful Skills/Tricks
* `jupyter notebook --no-browser --port=8080`  
`ssh -L 8080:localhost:8720 sunyingge@172.17.128.236 ssh -N -L 8720:localhost:8080 node1`  
You can think of this as running the first SSH and tell it to run another SSH on the server immediately after login. `8080:localhost:8180` is like address/port assigning (`8080=localhost:8180`). That is, your local 8080 port now points to the 8180 port of the remote machine so when you chain them together the local `8080` port is basically `localhost:localhost:8080`. The `-N` is basically equvalent telling SSH to connect but do nothing more and since you tell the first SSH to run another SSH client, only the second SSH command should have the `-N` option enabled.
* `tmux`, a particularly useful tool when you are working on a server, look it up.
* Usefull snippet:  
    ```python
    if os.path.exists(args.pkl_dir):
        input("Result file already exists. Press enter to continue and overwrite it...")
    ```
* `%matplotlib inline` - The magic.  
`%matplotlib notebook` This is not as recommended since it can cause weird ass issues.
* Python dependencies/package directory: `/rdfs/data/python_shared/Linux`.

## Linux Cheatsheet
* Count number of files recursively: `find $DIR -type f | wc -l`
* Search: `find /rdfs/fast/home/sunyingge/anaconda3/envs/tf14/lib/python3.6/site-packages/tensorflow/ -name tf_upgrade_v2*` (additionally you can specify `-type d` and `-print`)  
 Name example: `"*$STR*"`
* `tar -czvf name_of_archive.tar.gz $DIR && rm -rf $DIR`: compress and remove a large number of files.
* `ls -l --block-size=M/G $DIR`: File size.
* `du -h --max-depth=0 $DIR`: Folder size.
# Ideas
* Regularization: weight decay/l2 norm loss  
tensorflow.org/guide/migrate
* Use the upgrade script to only upgrade the model module and write a new pipeline?
* Loss weight to speed up training?

# Notes
* Tensorflow official low-level multi-GPU example: github.com/tensorflow/models/tree/r1.10.0/tutorials/image/cifar10
* tensorpack-0.10.1 dependencies:
    * msgpack-1.0.0
    * tabulate-0.8.7
    * msgpack-numpy-0.4.6

# Garage
What need to be changed in `test_graph_disease.py`:  
`info_paths`  
CONFIG["default007"], TENSOR_NAME  
target_dir of `pickle.dump()`  
import path

`demon.py` for data preparation.

python train.py sess_eval /rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_0613_1205_20/UNet_0611.json --batch_size 16 --eval_multi --model_folder /rdfs/fast/home/sunyingge/data/models/workdir_0611/SEResUNET_0613_1205_20/ --model_list model-173272 --gpus_to_use 0
