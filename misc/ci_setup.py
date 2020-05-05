print("* Taichi Installer")

import os
import sys
import platform
import subprocess
from os import environ
import platform

print(platform.architecture())
build_type = 'default'

# Utils

import struct

assert struct.calcsize(
    'P'
) * 8 == 64, "Only 64-bit platforms are supported. Current platform: {}".format(
    struct.calcsize('P') * 8)

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    print("\nPlease restart with python3. \n(Taichi supports Python 3.5+)\n")
    print("Current version:", sys.version_info)
    exit(-1)

print(
    "Note: this is the setup script for the taichi compiler DEVELOPERS. Language users please use pip to install the python wheels."
)


def get_python_executable():
    return sys.executable.replace('\\', '/')


def get_shell_name():
    return environ['SHELL'].split('/')[-1]


def get_shell_rc_name():
    shell = get_shell_name()
    if shell == 'bash':
        return '~/.bashrc'
    elif shell == 'zsh':
        return '~/.zshrc'
    else:
        assert False, 'No shell rc file specified for shell "{}"'.format(shell)


def get_username():
    if build_type == 'ci':
        os.environ['TI_CI'] = '1'
        username = 'travis'
    else:
        assert get_os_name() != 'win'
        import pwd
        username = pwd.getpwuid(os.getuid())[0]
    return username


def check_command_existence(cmd):
    return os.system('type {}'.format(cmd)) == 0


def execute_command(line, allow_nonzero_output=0):
    print('Executing command:', line)
    ret = os.system(line)
    if ret and not allow_nonzero_output:
        print('failed.')
        exit(-1)


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


def get_default_directory_name():
    '''
  osname = get_os_name()
  if osname == 'linux':
    username = get_username()
    return '/home/{}/repos'.format(username)
  elif osname == 'osx':
    username = get_username()
    return '/Users/{}/repos'.format(username)
  else:
    #Windows
    return os.getcwd()
  '''
    return os.getcwd()


def append_to_shell_rc(line):
    if get_os_name() != 'win':
        execute_command('echo "{}" >> {}'.format(line, get_shell_rc_name()))
    else:
        print(
            "Warning: Windows environment variable persistent edits are not supported"
        )


def set_env(key, val, val_now=None):
    if val_now is None:
        val_now = val
    val = str(val)
    val_now = str(val_now)
    append_to_shell_rc("export {}={}".format(key, val))
    os.environ[key] = val_now


def get_path_separator():
    if get_os_name() == 'win':
        return ';'
    else:
        return ':'


def test_installation():
    return subprocess.run(
        [get_python_executable(), "-c", "import taichi as tc"]).returncode == 0


# (Stateful) Installer class


class Installer:
    def __init__(self):
        # parser = argparse.ArgumentParser()
        # parser.parse_args()
        self.build_type = None

    def setup_repo(self):
        cwd = os.getcwd()
        print("Current directory:", cwd)

        execute_command("git submodule update --init --recursive")

    def run(self):
        assert get_os_name() in ['linux', 'osx', 'win'], \
          'Platform {} is not currently supported by this script. Please install manually.'.format(
            get_os_name())
        if len(sys.argv) > 1:
            self.build_type = sys.argv[1]
            print('Build type: ', self.build_type)
        else:
            self.build_type = 'default'
        global build_type
        build_type = self.build_type

        print('Build type = {}'.format(self.build_type))

        assert self.build_type in ['default', 'ci']

        check_command_existence('wget')
        try:
            import pip
            print('pip3 installation detected')
        except Exception as e:
            print(e)
            print('Installing pip3')
            execute_command('wget https://bootstrap.pypa.io/get-pip.py')
            subprocess.run([get_python_executable(), "get-pip.py", "--user"])
            execute_command('rm get-pip.py')

        subprocess.run([
            get_python_executable(),
            "-m",
            "pip",
            "install",
            "--user",
            "colorama",
            "numpy",
            "Pillow",
            "scipy",
            "pybind11",
            "GitPython",
            "yapf",
            "distro",
            "pytest",
            "autograd",
            "astor",
            "pytest-xdist",
            "pytest-rerunfailures",
        ])
        print("importing numpy test:")
        ret = subprocess.run(
            [get_python_executable(), "-c", "import numpy as np"])
        print("ret:", ret)

        execute_command('cmake --version')
        if get_os_name() == 'osx':
            # Check command existence
            check_command_existence('git')
            check_command_existence('cmake')
        elif get_os_name() == 'linux':
            check_command_existence('sudo')
            # TODO: this works for Ubuntu only
            if self.build_type != 'ci':
                import distro
                dist = distro.id()
            else:
                dist = 'ubuntu'
            print("Linux distribution '{}' detected".format(dist))
            if dist == 'ubuntu':
                if self.build_type != 'ci':  # Currently the CI machines have no sudo
                    execute_command('sudo apt-get update')
                    if self.build_type == 'ci':
                        execute_command(
                            'sudo apt-get install -y python3-dev libx11-dev')
                    else:
                        execute_command(
                            'sudo apt-get install -y python3-dev git build-essential cmake make g++ libx11-dev'
                        )
            elif dist == 'arch':
                execute_command('sudo pacman --needed -S git cmake make gcc')
            elif dist == 'fedora':
                execute_command(
                    'sudo dnf install python3-devel git cmake libX11-devel')
            else:
                print("Unsupported Linux distribution.")

        subprocess.run([
            get_python_executable(), "-m", "pip", "install", "--user", "psutil"
        ])

        self.setup_repo()

        # TODO: Make sure there is no existing Taichi ENV
        if self.build_type != 'ci':
            self.repo_dir = os.getcwd()
            set_env('TAICHI_REPO_DIR', self.repo_dir)

            set_env(
                'PYTHONPATH', '\$TAICHI_REPO_DIR/python/' +
                get_path_separator() + '\$PYTHONPATH',
                '{}/python/'.format(self.repo_dir) + get_path_separator() +
                os.environ.get('PYTHONPATH', ''))
            set_env(
                'PATH',
                '\$TAICHI_REPO_DIR/bin/' + get_path_separator() + '\$PATH',
                os.path.join(self.repo_dir, 'bin') + get_path_separator() +
                os.environ.get('PATH', ''))

            os.environ['PYTHONIOENCODING'] = 'utf-8'
            print('PYTHONPATH={}'.format(os.environ['PYTHONPATH']))

            execute_command('echo $PYTHONPATH')
        elif get_os_name() != 'win':
            # compile ..
            os.makedirs('build', exist_ok=True)
            execute_command('cd build && cmake .. -DTI_WITH_CUDA:BOOL=OFF')
            execute_command('cd build && make -j 10')
        return
        if test_installation():
            print('  Successfully Installed Taichi at {}.'.format(
                self.repo_dir))
            if get_os_name() != 'win':
                if execute_command('ti') != 0:
                    print('  Warning: shortcut "ti" does not work.')
                print('  Please execute')
                print('    source ', get_shell_rc_name())
                print('  or restart your terminal.')
        else:
            print('  Error: installation failed.')
            exit(-1)


if __name__ == '__main__':
    installer = Installer()
    installer.run()
