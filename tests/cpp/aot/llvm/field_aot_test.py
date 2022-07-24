import argparse

from utils import compile_field_aot

import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str)
parser.add_argument("--cgraph", action='store_true', default=False)
args = parser.parse_args()

if __name__ == "__main__":
    compile_for_cgraph = args.cgraph
    if args.arch == "cpu":
        compile_field_aot(arch=ti.cpu, compile_for_cgraph=compile_for_cgraph)
    elif args.arch == "cuda":
        compile_field_aot(arch=ti.cuda, compile_for_cgraph=compile_for_cgraph)
    else:
        assert False
