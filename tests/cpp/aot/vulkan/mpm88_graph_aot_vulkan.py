import os
import sys

from mpm88_graph_aot import compile_mpm88_graph

import taichi as ti

compile_mpm88_graph(arch=ti.vulkan)
