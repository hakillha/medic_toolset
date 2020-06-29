import os, shutil, time
from os.path import join as pj

def gen_outdirs(args, pipeline, cfg_file, train_debug):
    """
        args:
            args: Requires "resume" and "output_dir".
            pipeline: Type of the pipeline, ["tp", "tf_data"].
    """
    # if a ckpt dir is provided and exists, it's set to be output dir
    if args.resume != None and os.path.exists(os.path.dirname(args.resume)):
        if pipeline == "tp" and args.resume_epoch == 1:
            input("You are resuming but the epoch number is 1, press Enter to continue if you're finetuning...")
        output_dir = os.path.dirname(args.resume)
    elif not os.path.exists(args.output_dir):
        output_dir = args.output_dir + time.strftime("%m%d_%H%M_%S", time.localtime())
        os.makedirs(output_dir)
    else:
        output_dir = args.output_dir
    out_res_dir = pj(output_dir, "result")
    if not os.path.exists(out_res_dir):
        os.makedirs(out_res_dir)
    # Avoid overwritting config file. This is not performed under debug mode.
    if not train_debug:
        if os.path.exists(pj(output_dir, os.path.basename(cfg_file))):
            input("Config file will NOT be overwritten. Press Enter to continue...")
        else:
            shutil.copy(cfg_file, output_dir)
    outdirs = dict(output_dir=output_dir, out_res_dir=out_res_dir)
    return outdirs