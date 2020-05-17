# Ideas
* Regularization: weight decay  
tensorflow.org/guide/migrate
* Momentum
* Use the upgrade script to only upgrade the model module and write a new pipeline?

# Note
* Check files integrity and shape consistency between images and annoataions.

# Training
> Training procedures:  
> 1. `fast/cf_mod/demon.py` for data preparation.
> 2. `tools/train_tf_lowlevel.py`
> 3. `tools/evaluate.py`
## `train_tf_lowlevel.py`
TODO: Add a specific name to ease the lookup for the input/output tensors.

# TODO
* Save tensorboard log
* Keep a separate text file to keep a good track of all the files in each split set and to keep everyone's data consistent.

# Linux Cheatsheet
* Count number of files recursively: `find DIR_NAME -type f | wc -l`

## Useful Skills/Tricks
* `ssh -L 8080:localhost:8180 sunyingge@172.17.128.236 ssh -N -L 8180:localhost:8080 node2`  
You can think of this as running the first SSH and tell it to run another SSH on the server immediately after login. `8080:localhost:8180` is like address/port assigning (`8080=localhost:8180`). That is, your local 8080 port now points to the 8180 port of the remote machine so when you chain them together the local `8080` port is basically `localhost:localhost:8080`. The `-N` is basically equvalent telling SSH to connect but do nothing more and since you tell the first SSH to run another SSH client, only the second SSH command should have the `-N` option enabled.
* `tmux`, a particularly useful tool when you are working on a server, look it up.
* Usefull snippet:  
    ```python
    if os.path.exists(args.pkl_dir):
        input("Result file already exists. Press enter to continue and overwrite it...")
    ```

# Garage
What need to be changed in `test_graph_disease.py`:  
`info_paths`  
CONFIG["default007"], TENSOR_NAME  
target_dir of `pickle.dump()`  
import path

`demon.py` for data preparation.