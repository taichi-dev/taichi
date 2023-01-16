import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path


def is_in_venv() -> bool:
    '''
    Are we in a virtual environment?
    '''
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix')
                                           and sys.base_prefix != sys.prefix)


def get_cache_home() -> Path:
    '''
    Get the cache home directory. All intermediate files should be stored here.
    '''
    if platform.system() == 'Windows':
        return Path(os.environ['LOCALAPPDATA']) / 'build-cache'
    else:
        return Path.home() / '.cache' / 'build-cache'


def ensure_dependencies(fn='requirements.txt'):
    '''
    Automatically install dependencies if they are not installed.
    '''

    p = Path(__file__).parent.parent / fn
    if not p.exists():
        raise RuntimeError(f'Cannot find {p}')

    bootstrap_root = get_cache_home() / 'bootstrap'
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bootstrap_root))

    user = '' if is_in_venv() else '--user'

    with open(p) as f:
        deps = [i.strip().split('=')[0] for i in f.read().splitlines()]

    try:
        for dep in deps:
            importlib.import_module(dep)
    except ModuleNotFoundError:
        print('Installing dependencies...')
        if os.system(
                f'{sys.executable} -m pip install {user} -U pip setuptools'):
            raise Exception('Unable to upgrade pip!')
        if os.system(
                f'{sys.executable} -m pip install {user} -U -r {p} --target={bootstrap_root}'
        ):
            raise Exception('Unable to install dependencies!')

        if platform.system != 'Windows':
            # GitHub Actions will treat the step as completed when doing os.execl in Windows,
            # since Windows does not have real execve, its behavior is emulated by spawning a new process and
            # terminating the current process. So we do not use os.execl in Windows.

            # use subprocess.Popen to execute self
            proc = subprocess.Popen([sys.executable, *sys.argv])
            proc.wait()
        else:
            os.execl(sys.executable, sys.executable, *sys.argv)


def chdir_to_root():
    '''
    Change working directory to the root of the repository
    '''
    root = Path('/')
    p = Path(__file__).resolve()
    while p != root:
        if (p / 'setup.py').exists():
            os.chdir(p)
            break
        p = p.parent


def set_common_env():
    '''
    Set common environment variables.
    '''
    os.environ['TI_CI'] = '1'


_Environ = os.environ.__class__


class _EnvironWrapper(_Environ):
    def __setitem__(self, name: str, value: str) -> None:
        orig = self.get(name, None)
        _Environ.__setitem__(self, name, value)
        new = self[name]

        from .escapes import escape_codes

        G = escape_codes['bold_green']
        R = escape_codes['bold_red']
        N = escape_codes['reset']

        if orig == new:
            pass
        elif orig == None:
            print(f'{G}:: ENV+ {name}={new}{N}', file=sys.stderr, flush=True)
        elif new.startswith(orig):
            l = len(orig)
            print(f'{G}:: ENV{N} {name}={new[:l]}{G}{new[l:]}{N}',
                  file=sys.stderr,
                  flush=True)
        elif new.endswith(orig):
            l = len(new) - len(orig)
            print(f'{G}:: ENV{N} {name}={G}{new[:l]}{N}{new[l:]}',
                  file=sys.stderr,
                  flush=True)
        else:
            print(f'{R}:: ENV- {name}={orig}{N}', file=sys.stderr, flush=True)
            print(f'{G}:: ENV+ {name}={new}{N}', file=sys.stderr, flush=True)


def monkey_patch_environ():
    '''
    Monkey patch os.environ to print changes.
    '''
    os.environ.__class__ = _EnvironWrapper


def early_init():
    '''
    Do early initialization.
    This must be called before any other non-stdlib imports.
    '''
    ensure_dependencies()
    chdir_to_root()
    monkey_patch_environ()
    set_common_env()
