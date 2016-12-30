import os

default_num_threads = 8

def get_num_cores():
    return os.environ.get('TAICHI_NUM_THREADS', default_num_threads)

def get_output_directory():
    return os.environ.get('TAICHI_OUTPUT_DIR', '../taichi_output')
