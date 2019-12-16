import os
import multiprocessing

from taichi.misc.util import get_os_name

default_num_threads = multiprocessing.cpu_count()


def get_num_cores():
  return os.environ.get('TAICHI_NUM_THREADS', default_num_threads)


def get_directory(dir):
  return os.path.join(get_repo_directory(), *dir.split('/'))


def get_repo_directory():
  if 'TAICHI_REPO_DIR' not in os.environ:
    return os.path.join(os.environ.get('HOME'), ".taichi")
  return os.environ.get('TAICHI_REPO_DIR')


def get_project_directory(project=None):
  if project:
    return os.path.join(get_project_directory(), project)
  else:
    return os.path.join(get_repo_directory(), 'projects')


def get_runtime_directory():
  bin_rel_path = ['extneral', 'lib']
  return os.environ.get('TAICHI_BIN_DIR',
                        os.path.join(get_repo_directory(), *bin_rel_path))


def get_build_directory():
  bin_rel_path = ['build']
  return os.environ.get('TAICHI_BIN_DIR',
                        os.path.join(get_repo_directory(), *bin_rel_path))


def get_bin_directory():
  if get_os_name() == 'win':
    # for the dlls
    bin_rel_path = ['runtimes']
  else:
    bin_rel_path = ['build']
  return os.path.join(get_repo_directory(), *bin_rel_path)


def get_output_directory():
  return os.environ.get('TAICHI_OUTPUT_DIR',
                        os.path.join(get_repo_directory(), 'outputs'))


def get_output_path(path, create=False):
  path = os.path.join(get_output_directory(), path)
  if create:
    os.makedirs(path, exist_ok=True)
  return path


def get_asset_directory():
  asset_dir = os.environ.get('TAICHI_ASSET_DIR', '').strip()
  if asset_dir == '':
    return os.path.join(get_repo_directory(), 'assets')
  else:
    return asset_dir


def get_asset_path(path, *args):
  return os.path.join(get_asset_directory(), path, *args)
