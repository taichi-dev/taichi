from utils import compile_graph_aot

import taichi as ti

if __name__ == "__main__":
    compile_graph_aot(arch=ti.cuda)
