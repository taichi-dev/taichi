import os
import sys

import taichi as ti

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mpm88_graph_aot import compile_mpm88_graph

compile_mpm88_graph(arch=ti.cpu)
