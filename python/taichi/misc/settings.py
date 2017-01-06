import os
from taichi.misc.util import get_os_name

default_num_threads = 2


def get_num_cores():
    return os.environ.get('TAICHI_NUM_THREADS', default_num_threads)


def get_root_directory():
    return os.environ['TAICHI_ROOT_DIR'] + '/'


def get_bin_directory():
    if get_os_name() == 'win':
        # for the dlls
        bin_rel_path = '/taichi/runtimes/'
    else:
        bin_rel_path = '/taichi/build/'
    return os.environ.get('TAICHI_BIN_DIR', get_root_directory() + bin_rel_path) + '/'


def get_output_directory():
    return os.environ.get('TAICHI_OUTPUT_DIR', get_root_directory() + '/taichi_outputs/') + '/'


def get_output_path(path):
    return '/'.join([get_output_directory(), path])


def get_asset_directory():
    return os.environ.get('TAICHI_ASSET_DIR', get_root_directory() + '/taichi_assets/') + '/'


def get_asset_path(path):
    return '/'.join([get_asset_directory(), path])
