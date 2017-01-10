import atexit
import os
import shutil
import sys

from taichi.misc.settings import get_output_directory, get_bin_directory, get_root_directory
from taichi.misc.util import get_os_name, get_unique_task_id

CREATE_SAND_BOX_ON_WINDOWS = True

if get_os_name() == 'osx':
    bin_dir = get_bin_directory() + '/'
    if os.path.exists(bin_dir + 'libtaichi_core.dylib'):
        tmp_cwd = os.getcwd()
        os.chdir(bin_dir)
        shutil.copy('libtaichi_core.dylib', 'taichi_core.so')
        sys.path.append(bin_dir)
        import taichi_core as tc_core

        os.chdir(tmp_cwd)
    else:
        assert False, "Library taichi_core doesn't exist."
elif get_os_name() == 'linux':
    bin_dir = get_bin_directory() + '/'
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/'
    if os.path.exists(bin_dir + 'libtaichi_core.so'):
        tmp_cwd = os.getcwd()
        os.chdir(bin_dir)
        sys.path.append(bin_dir)
        shutil.copy('libtaichi_core.so', 'taichi_core.so')
        import taichi_core as tc_core

        os.chdir(tmp_cwd)
    else:
        assert False, "Library taichi_core doesn't exist."
    if os.path.exists('libtaichi_core.so'):
        shutil.copy('libtaichi_core.so', 'taichi_core.so')
        sys.path.append(".")
        import taichi_core as tc_core
elif get_os_name() == 'win':
    bin_dir = get_bin_directory() + '/'
    dll_path = bin_dir + '/Release/taichi_core.dll'

    # The problem here is, on windows, when an dll/pyd is loaded, we can not write to it any more...

    # Ridiculous...
    old_wd = os.getcwd()
    os.chdir(bin_dir)

    if os.path.exists(dll_path):
        if CREATE_SAND_BOX_ON_WINDOWS:
            # So let's just create a sandbox for separated core lib development and loading
            dir = get_output_directory() + '/tmp/' + get_unique_task_id() + '/'
            os.makedirs(dir)
            '''
            for fn in os.listdir(bin_dir):
                if fn.endswith('.dll') and fn != 'taichi_core.dll':
                    print dir + fn, bin_dir + fn
                    # Why can we create symbolic links....
                    # if not ctypes.windll.kernel32.CreateSymbolicLinkW(bin_dir + fn, dir + fn, 0):
                    #    raise OSError
                    shutil.copy(bin_dir + fn, dir + fn)
            '''
            shutil.copy(dll_path, dir + 'taichi_core.pyd')
            sys.path.append(dir)
        else:
            shutil.copy(dll_path, bin_dir + 'taichi_core.pyd')
            sys.path.append(bin_dir)
    else:
        assert False, "Library taichi_core doesn't exist."
    import taichi_core as tc_core

    os.chdir(old_wd)

def at_startup():
    assert os.path.exists(get_root_directory()), 'Please make sure $TAICHI_ROOT_DIR [' + get_root_directory() + '] exists.'
    output_dir = get_output_directory()
    if not os.path.exists(output_dir):
        print 'Making output directory'
        os.mkdir(output_dir)

at_startup()

@atexit.register
def clean_libs():
    pass
