import os
import re
import shutil
import sys
import ctypes
import warnings
from pathlib import Path
from colorama import Fore, Back, Style
from taichi.misc.settings import get_output_directory, get_build_directory, get_bin_directory, get_repo_directory, get_runtime_directory
from taichi.misc.util import get_os_name, get_unique_task_id

if sys.version_info[0] < 3 or sys.version_info[1] <= 5:
    print("\nPlease restart with Python 3.6+\n")
    print("Current Python version:", sys.version_info)
    exit(-1)

warnings.filterwarnings('always')

ti_core = None


def in_docker():
    if os.environ.get("TI_IN_DOCKER", "") == "":
        return False
    else:
        return True


def import_ti_core(tmp_dir=None):
    global ti_core
    if get_os_name() != 'win':
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(258)  # 258 = RTLD_NOW | RTLD_GLOBAL
    else:
        pyddir = os.path.join(package_root(), 'lib')
        os.environ['PATH'] += ';' + pyddir
    try:
        import taichi_core as core
    except Exception as e:
        if isinstance(e, ImportError):
            print(
                Fore.YELLOW + "Share object taichi_core import failed, "
                "check this page for possible solutions:\n"
                "https://taichi.readthedocs.io/en/stable/install.html#troubleshooting"
                + Fore.RESET)
        raise e
    ti_core = core
    if get_os_name() != 'win':
        sys.setdlopenflags(old_flags)
    lib_dir = os.path.join(package_root(), 'lib')
    core.set_lib_dir(locale_encode(lib_dir))
    if tmp_dir is not None:
        core.set_tmp_dir(locale_encode(tmp_dir))


def locale_encode(s):
    try:
        import locale
        encoding = locale.getdefaultlocale()[1]
    except:
        encoding = 'utf8'
    return s.encode(encoding)


def is_ci():
    return os.environ.get('TI_CI', '') == '1'


def package_root():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')


def is_release():
    return os.environ.get('TAICHI_REPO_DIR', '') == ''


def get_core_shared_object():
    if is_release():
        directory = os.path.join(package_root(), 'lib')
    else:
        directory = get_bin_directory()
    return os.path.join(directory, 'libtaichi_core.so')


def get_repo():
    from git import Repo
    repo = Repo(get_repo_directory())
    return repo


def print_red_bold(*args, **kwargs):
    print(Fore.RED + Style.BRIGHT, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')


def has_suffix(f, suffixes):
    for suf in suffixes:
        if f.endswith('.' + suf):
            return True
    return False


def format_plain_text(fn):
    formatted = ''
    with open(fn, 'r') as f:
        for l in f:
            l = l.rstrip()
            if l.find('\t') != -1:
                print(f'Warning: found tab in {fn}. Skipping...')
                return
            formatted += l + '\n'
    while len(formatted) and formatted[-1] == '\n':
        formatted = formatted[:-1]
    formatted += '\n'
    with open(fn, 'w') as f:
        f.write(formatted)


def _find_clang_format_bin():
    candidates = ['clang-format-6.0', 'clang-format']
    result = None

    import subprocess as sp
    for c in candidates:
        try:
            if sp.run([c, '--version'], stdout=sp.DEVNULL,
                      stderr=sp.DEVNULL).returncode == 0:
                result = c
                break
        except:
            pass
    import colorama
    colorama.init()
    if result is None:
        print(Fore.YELLOW +
              'Did not find any clang-format executable, skipping C++ files',
              file=sys.stderr)
    else:
        print('C++ formatter: {}{}'.format(Fore.GREEN, result))
    print(Style.RESET_ALL)
    return result


def format(all=False, diff=None):
    import os
    import taichi as tc
    from yapf.yapflib.yapf_api import FormatFile
    repo = get_repo()

    if all:
        directories = [
            'taichi', 'tests', 'examples', 'misc', 'python', 'benchmarks',
            'docs', 'misc'
        ]
        files = []
        for d in directories:
            files += list(
                Path(os.path.join(tc.get_repo_directory(), d)).rglob('*'))
    else:
        if diff is None:

            def find_diff_or_empty(s):
                try:
                    return repo.index.diff(s)
                except:
                    return []

            # TODO(#628): Have a way to customize the repo names, in order to
            # support noncanonical namings.
            #
            # Finds all modified files from upstream/master to working tree
            # 1. diffs between the index and upstream/master. Also inclulde
            # origin/master for repo owners.
            files = find_diff_or_empty('upstream/master')
            files += find_diff_or_empty('origin/master')
            # 2. diffs between the index and the working tree
            # https://gitpython.readthedocs.io/en/stable/tutorial.html#obtaining-diff-information
            files += repo.index.diff(None)
        else:
            files = repo.index.diff(diff)
        files = list(
            map(lambda x: os.path.join(tc.get_repo_directory(), x.a_path),
                files))

    files = sorted(set(map(str, files)))
    clang_format_bin = _find_clang_format_bin()
    print('Code formatting ...')
    for fn in files:
        if not os.path.exists(fn):
            continue
        if os.path.isdir(fn):
            continue
        if fn.find('.pytest_cache') != -1:
            continue
        if fn.find('docs/build/') != -1:
            continue
        if re.match(r'.*examples\/[a-z_]+\d\d+\.py$', fn):
            print(f'Skipping example file {fn}...')
            continue
        if fn.endswith('.py'):
            print('Formatting "{}"'.format(fn))
            FormatFile(fn,
                       in_place=True,
                       style_config=os.path.join(tc.get_repo_directory(),
                                                 'misc', '.style.yapf'))
        elif clang_format_bin and has_suffix(fn, ['cpp', 'h', 'cu', 'cuh']):
            print('Formatting "{}"'.format(fn))
            os.system('{} -i -style=file {}'.format(clang_format_bin, fn))
        elif has_suffix(fn, ['txt', 'md', 'rst', 'cfg', 'll', 'ptx']):
            print('Formatting "{}"'.format(fn))
            format_plain_text(fn)
        elif has_suffix(fn, [
                'pyc', 'png', 'jpg', 'bmp', 'gif', 'gitignore', 'whl', 'mp4',
                'html'
        ]):
            pass
        else:
            print(f'Skipping {fn}...')

    print('Formatting done!')


create_sand_box_on_windows = True


def build():
    tmp_cwd = os.getcwd()
    bin_dir = get_build_directory()

    try:
        os.mkdir(bin_dir)
    except:
        pass
    os.chdir(bin_dir)

    import multiprocessing
    print('Building taichi...')
    num_make_threads = min(20, multiprocessing.cpu_count())
    if get_os_name() == 'win':
        make_ret = os.system(
            "msbuild /p:Configuration=Release /p:Platform=x64 /m taichi.sln")
    else:
        make_ret = os.system('make -j {}'.format(num_make_threads))
    if make_ret != 0:
        print('  Error: Build failed.')
        exit(-1)

    os.chdir(tmp_cwd)


def prepare_sandbox(src):
    global g_tmp_dir
    assert os.path.exists(src)
    import atexit
    import shutil
    from tempfile import mkdtemp
    tmp_dir = mkdtemp(prefix='taichi-')
    atexit.register(shutil.rmtree, tmp_dir)
    print(f'[Taichi] preparing sandbox at {tmp_dir}')
    dest = os.path.join(tmp_dir, 'taichi_core.so')
    shutil.copy(src, dest)
    os.mkdir(os.path.join(tmp_dir, 'runtime/'))
    return tmp_dir


if is_release():
    print("[Taichi] mode=release")
    sys.path.append(os.path.join(package_root(), 'lib'))
    if get_os_name() != 'win':
        link_src = os.path.join(package_root(), 'lib', 'taichi_core.so')
        link_dst = os.path.join(package_root(), 'lib', 'libtaichi_core.so')
        # For llvm jit to find the runtime symbols
        if not os.path.exists(link_dst):
            os.symlink(link_src, link_dst)
    import_ti_core()
    if get_os_name() != 'win':
        dll = ctypes.CDLL(get_core_shared_object(), mode=ctypes.RTLD_GLOBAL)

    ti_core.set_python_package_dir(package_root())
    os.makedirs(ti_core.get_repo_dir(), exist_ok=True)
else:
    print("[Taichi] mode=development")
    if get_os_name() == 'osx':
        bin_dir = get_bin_directory()
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = get_runtime_directory()
        lib_path = os.path.join(bin_dir, 'libtaichi_core.dylib')
        tmp_cwd = os.getcwd()
        tmp_dir = prepare_sandbox(lib_path)
        os.chdir(tmp_dir)
        sys.path.append(tmp_dir)
        import taichi_core as ti_core
        os.chdir(tmp_cwd)

    elif get_os_name() == 'linux':
        bin_dir = get_bin_directory()
        if 'LD_LIBRARY_PATH' in os.environ:
            os.environ['LD_LIBRARY_PATH'] += ':/usr/lib64/'
        else:
            os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/'
        lib_path = os.path.join(bin_dir, 'libtaichi_core.so')
        assert os.path.exists(lib_path)
        tmp_cwd = os.getcwd()
        tmp_dir = prepare_sandbox(lib_path)
        os.chdir(tmp_dir)
        sys.path.append(tmp_dir)
        try:
            import_ti_core(tmp_dir)
        except Exception as e:
            from colorama import Fore, Back, Style
            print_red_bold("Taichi core import failed: ", end='')
            print(e)
            exit(-1)
        os.chdir(tmp_cwd)

    elif get_os_name() == 'win':
        bin_dir = get_bin_directory()
        dll_path_invalid = os.path.join(bin_dir, 'libtaichi_core.dll')
        assert not os.path.exists(dll_path_invalid)

        possible_folders = ['Debug', 'RelWithDebInfo', 'Release']
        detected_dlls = []
        for folder in possible_folders:
            dll_path = os.path.join(bin_dir, folder, 'taichi_core.dll')
            if os.path.exists(dll_path):
                detected_dlls.append(dll_path)

        if len(detected_dlls) == 0:
            raise FileNotFoundError(
                f'Cannot find Taichi core dll under {get_bin_directory()}/{possible_folders}'
            )
        elif len(detected_dlls) != 1:
            print('Warning: multiple Taichi core dlls found:')
            for dll in detected_dlls:
                print(' ', dll)
            print(f'Using {detected_dlls[0]}')

        dll_path = detected_dlls[0]

        # On windows when an dll/pyd is loaded, we cannot write to it any more
        old_wd = os.getcwd()
        os.chdir(bin_dir)

        if create_sand_box_on_windows:
            # Create a sandbox for separated core lib development and loading
            folder = os.path.join(get_output_directory(), 'tmp',
                                  get_unique_task_id())

            lib_dir = os.path.join(get_repo_directory(), 'external', 'lib')
            os.environ['PATH'] += ';' + lib_dir

            os.makedirs(folder)
            shutil.copy(dll_path, os.path.join(folder, 'taichi_core.pyd'))
            os.environ['PATH'] += ';' + folder
            sys.path.append(folder)
        else:
            shutil.copy(dll_path, os.path.join(bin_dir, 'taichi_core.pyd'))
            sys.path.append(bin_dir)
        try:
            import taichi_core as ti_core
        except Exception as e:
            print(e)
            print()
            print(
                'Hint: please make sure the major and minor versions of the Python executable is correct.'
            )
            print()
            raise e

        os.chdir(old_wd)

log_level = os.environ.get('TI_LOG_LEVEL', '')
if log_level:
    ti_core.set_logging_level(log_level)


def get_dll_name(name):
    if get_os_name() == 'linux':
        return 'libtaichi_%s.so' % name
    elif get_os_name() == 'osx':
        return 'libtaichi_%s.dylib' % name
    elif get_os_name() == 'win':
        return 'taichi_%s.dll' % name
    else:
        assert False, "Unknown OS"


def load_module(name, verbose=True):
    if verbose:
        print('Loading module', name)
    try:
        if get_os_name() == 'osx':
            mode = ctypes.RTLD_LOCAL
        else:
            mode = ctypes.RTLD_GLOBAL
        if '.so' in name:
            ctypes.PyDLL(name, mode=mode)
        else:
            ctypes.PyDLL(os.path.join(get_repo_directory(), 'build',
                                      get_dll_name(name)),
                         mode=mode)
    except Exception as e:
        print(Fore.YELLOW +
              "Warning: module [{}] loading failed: {}".format(name, e) +
              Style.RESET_ALL)


def at_startup():
    if not is_release():
        output_dir = get_output_directory()
        if not os.path.exists(output_dir):
            print('Making output directory')
            os.mkdir(output_dir)

    ti_core.set_core_state_python_imported(True)


def start_memory_monitoring(output_fn, pid=-1, interval=1):
    # removing dependency on psutil
    return
    import os, psutil, time
    if pid == -1:
        pid = os.getpid()
    import multiprocessing

    def task():
        with open(output_fn, 'w') as f:
            process = psutil.Process(pid)
            while True:
                try:
                    mem = process.memory_info().rss
                except:
                    mem = -1
                time.sleep(interval)
                print(time.time(), mem, file=f)
                f.flush()

    proc = multiprocessing.Process(target=task, daemon=True)
    proc.start()


def require_version(major, minor=None, patch=None):
    versions = [
        int(ti_core.get_version_major()),
        int(ti_core.get_version_minor()),
        int(ti_core.get_version_patch()),
    ]
    match = major == versions[0] and (
        minor < versions[1] or minor == versions[1] and patch <= versions[2])
    if match:
        return
    else:
        print("Taichi version mismatch. required >= {}.{}.{}".format(
            major, minor, patch))
        print("Installed =", ti_core.get_version_string())
        raise Exception("Taichi version mismatch")


at_startup()


def _print_taichi_header():
    dev_mode = not is_release()

    header = '[Taichi] '
    if dev_mode:
        header += '<dev mode>, '
    else:
        header += f'version {ti_core.get_version_string()}, '

    supported_archs = ['cpu']
    if ti_core.with_cuda():
        supported_archs.append('cuda')
    if ti_core.with_opengl():
        supported_archs.append('opengl')
    if ti_core.with_metal():
        supported_archs.append('metal')
    if len(supported_archs) == 1:
        supported_archs[0] = 'cpu only'
    archs_str = ', '.join(sorted(supported_archs))
    header += f'supported archs: [{archs_str}], '

    llvm_version = ti_core.get_llvm_version_string()
    header += f'llvm {llvm_version}, '

    commit_hash = ti_core.get_commit_hash()
    commit_hash = commit_hash[:8]
    header += f'commit {commit_hash}, '

    py_ver = '.'.join(str(x) for x in sys.version_info[:3])
    header += f'python {py_ver}'

    print(header)


_print_taichi_header()
