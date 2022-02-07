import datetime
import functools
import json
import os

import jsbeautifier
from taichi._lib import core as ti_core

import taichi as ti


def get_benchmark_dir():
    return os.path.dirname(os.path.realpath(__file__))


def benchmark_async(func):
    @functools.wraps(func)
    def body():
        for arch in [ti.cpu, ti.cuda]:
            for async_mode in [True, False]:
                os.environ['TI_CURRENT_BENCHMARK'] = func.__name__
                ti.init(arch=arch,
                        async_mode=async_mode,
                        kernel_profiler=True,
                        verbose=False)
                if arch == ti.cpu:
                    scale = 2
                else:
                    # Use more data to hide compilation overhead
                    # (since CUDA runs much faster than CPUs)
                    scale = 64
                func(scale)

    return body


def dump2json(obj):
    obj2dict = obj if type(obj) is dict else obj.__dict__
    options = jsbeautifier.default_options()
    options.indent_size = 4
    return jsbeautifier.beautify(json.dumps(obj2dict), options)


def datatime_with_format():
    return datetime.datetime.now().isoformat()


def get_commit_hash():
    return ti_core.get_commit_hash()
