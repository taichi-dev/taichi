import argparse

from utils import compile_field_aot

import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    if args.arch == "cpu":
        compile_field_aot(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_field_aot(arch=ti.cuda)
    else:
        assert False
