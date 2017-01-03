import os

default_num_threads = 8


def get_num_cores():
    return os.environ.get('TAICHI_NUM_THREADS', default_num_threads)


def get_output_directory():
    return os.environ.get('TAICHI_OUTPUT_DIR', '../taichi_output')


def get_output_path(path):
    return '/'.join([get_output_directory(), path])


def get_asset_directory():
    return os.environ.get('TAICHI_ASSET_DIR', '../taichi_asset')


def get_asset_path(path):
    return '/'.join([get_asset_directory(), path])
