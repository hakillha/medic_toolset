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


```
batch_0505: 
defaultdict(<class 'int'>, {'covid': 771, 'thick': 1412, 'covid_thick': 720, 'thin': 81, 'covid_thin': 51, 'Not in covid nor normal_pneu': 864, 'normal': 722, 'normal_thick': 692, 'n
ormal_thin': 30})
batch_0507: 
defaultdict(<class 'int'>, {'covid': 152, 'thick': 85, 'covid_thick': 81, 'thin': 134, 'covid_thin': 71, 'normal': 67, 'normal_thin': 63, 'normal_thick': 4})
batch_0508: 
defaultdict(<class 'int'>, {'covid': 106, 'thick': 25, 'covid_thick': 25, 'thin': 81, 'covid_thin': 81})
batch_0509: 
defaultdict(<class 'int'>, {'normal': 163, 'thick': 163, 'normal_thick': 163})
batch_0510: 
defaultdict(<class 'int'>, {'covid': 233, 'thick': 227, 'covid_thick': 195, 'thin': 61, 'covid_thin': 38, 'normal': 55, 'normal_thin': 23, 'normal_thick': 32})
batch_0511: 
defaultdict(<class 'int'>, {'covid': 365, 'thin': 434, 'covid_thin': 123, 'thick': 670, 'covid_thick': 242, 'normal': 739, 'normal_thin': 311, 'normal_thick': 428})
batch_0513: 
defaultdict(<class 'int'>, {'covid': 147, 'thin': 133, 'covid_thin': 133, 'thick': 14, 'covid_thick': 14})
batch_0514: 
defaultdict(<class 'int'>, {'covid': 28, 'thick': 53, 'covid_thick': 28, 'normal': 54, 'thin': 29, 'normal_thin': 29, 'normal_thick': 25})
batch_0515: 
defaultdict(<class 'int'>, {'covid': 62, 'thick': 106, 'covid_thick': 61, 'thin': 19, 'covid_thin': 1, 'normal': 63, 'normal_thick': 45, 'normal_thin': 18})
```