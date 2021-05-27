import multiprocessing
import os
import platform


def get_num_cores():
    default_num_threads = multiprocessing.cpu_count()
    return os.environ.get('TAICHI_NUM_THREADS', default_num_threads)


def get_os_name():
    name = platform.platform()
    # in python 3.8, platform.platform() uses mac_ver() on macOS
    # it will return 'macOS-XXXX' instead of 'Darwin-XXXX'
    if name.lower().startswith('darwin') or name.lower().startswith('macos'):
        return 'osx'
    elif name.lower().startswith('windows'):
        return 'win'
    elif name.lower().startswith('linux'):
        return 'linux'
    assert False, "Unknown platform name %s" % name


def get_directory(dir):
    return os.path.join(get_repo_directory(), *dir.split('/'))


def get_repo_directory():
    if 'TAICHI_REPO_DIR' not in os.environ:
        repo_dir = os.path.join(os.environ.get('HOME'), ".taichi")
    else:
        repo_dir = os.environ.get('TAICHI_REPO_DIR')
        if not os.path.exists(repo_dir):
            raise ValueError(f"TAICHI_REPO_DIR [{repo_dir}] does not exist.")
    return repo_dir


def get_project_directory(project=None):
    if project:
        return os.path.join(get_project_directory(), project)
    else:
        return os.path.join(get_repo_directory(), 'projects')


def get_runtime_directory():
    bin_rel_path = ['external', 'lib']
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


__all__ = [
    'get_output_directory',
    'get_build_directory',
    'get_bin_directory',
    'get_repo_directory',
    'get_runtime_directory',
    'get_os_name',
]
