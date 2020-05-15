# bunch of random tools
import argparse, os, sys
import pickle

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("mode", choices=[])
    parser.add_argument("input_file")
    return parser.parse_args()

args = parse_args()

if args.mode == ""