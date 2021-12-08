# Optional environment variables supported by setup.py:
#   DEBUG
#     build the C++ taichi_core extension with debug symbols.
#
#   TAICHI_CMAKE_ARGS
#     extra cmake args for C++ taichi_core extension.

import glob
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
from distutils.command.clean import clean
from distutils.dir_util import remove_tree

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Software Development :: Compilers',
    'Topic :: Multimedia :: Graphics',
    'Topic :: Games/Entertainment :: Simulation',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]

project_name = os.getenv('PROJECT_NAME', 'taichi')
TI_VERSION_MAJOR = 0
TI_VERSION_MINOR = 8
TI_VERSION_PATCH = 7
version = f'{TI_VERSION_MAJOR}.{TI_VERSION_MINOR}.{TI_VERSION_PATCH}'

data_files = glob.glob('python/lib/runtime/*')
print(data_files)
packages = find_packages('python')
print(packages)

# Our python package root dir is python/
package_dir = 'python'

root_dir = os.path.abspath(os.path.dirname(__file__))


def get_python_executable():
    return sys.executable.replace('\\', '/')


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
    elif 'bsd' in name.lower():
        return 'unix'
    assert False, "Unknown platform name %s" % name


def remove_tmp(taichi_dir):
    shutil.rmtree(os.path.join(taichi_dir, 'assets'), ignore_errors=True)


def remove_files_with_extension(dir_name, extension):
    for file in os.listdir(dir_name):
        if file.endswith(extension):
            os.remove(os.path.join(dir_name, file))


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class EggInfo(egg_info):
    def run(self):
        taichi_dir = os.path.join(package_dir, 'taichi')
        remove_tmp(taichi_dir)

        shutil.copytree('external/assets', os.path.join(taichi_dir, 'assets'))

        egg_info.run(self)


# python setup.py build runs the following commands in order:
#   python setup.py build_py
#   python setup.py build_ext
class BuildPy(build_py):
    def run(self):
        build_py.run(self)
        taichi_dir = os.path.join(package_dir, 'taichi')
        remove_tmp(taichi_dir)


class CMakeBuild(build_ext):
    def parse_cmake_args_from_env(self):
        # Source: TAICHI_CMAKE_ARGS=... python setup.py ...
        import shlex
        cmake_args = os.getenv('TAICHI_CMAKE_ARGS', '')
        return shlex.split(cmake_args.strip())

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = root_dir
        self.build_temp = os.path.join(cmake_list_dir, 'build')

        build_directory = os.path.abspath(self.build_temp)

        cmake_args = self.parse_cmake_args_from_env()

        cmake_args += [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_directory}',
            f'-DPYTHON_EXECUTABLE={get_python_executable()}',
            f'-DTI_VERSION_MAJOR={TI_VERSION_MAJOR}',
            f'-DTI_VERSION_MINOR={TI_VERSION_MINOR}',
            f'-DTI_VERSION_PATCH={TI_VERSION_PATCH}',
        ]

        if shutil.which('ninja'):
            cmake_args += ['-GNinja']

        self.debug = os.getenv('DEBUG', '0') in ('1', 'ON')
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Assuming Makefiles
        if get_os_name() != 'win':
            num_threads = os.getenv('BUILD_NUM_THREADS',
                                    multiprocessing.cpu_count())
            build_args += ['--', f'-j{num_threads}']

        self.build_args = build_args

        env = os.environ.copy()
        os.makedirs(self.build_temp, exist_ok=True)

        print('-' * 10, 'Running CMake prepare', '-' * 40)
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                              cwd=self.build_temp,
                              env=env)

        print('-' * 10, 'Building extensions', '-' * 40)
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)

        self.prepare_package()

    def prepare_package(self):
        # We need to make sure these additional files are ready for
        #   - develop mode: must exist in local python/taichi/lib/ folder
        #   - install mode: must exist in self.build_lib/taichi/lib
        taichi_lib_dir = os.path.join('taichi', 'lib')
        for target in (
                os.path.join(package_dir, taichi_lib_dir),
                os.path.join(self.build_lib, taichi_lib_dir),
        ):
            runtime_dir = os.path.join(target, "runtime")
            core_dir = os.path.join(target, "core")
            os.makedirs(runtime_dir, exist_ok=True)

            if get_os_name() == 'linux' or get_os_name(
            ) == 'unix' or get_os_name() == 'osx':
                remove_files_with_extension(core_dir, ".so")
            else:
                remove_files_with_extension(core_dir, ".pyd")
            if get_os_name() == 'osx':
                remove_files_with_extension(runtime_dir, ".dylib")
            remove_files_with_extension(runtime_dir, ".bc")

            if get_os_name() == 'linux' or get_os_name() == 'unix':
                shutil.copy(os.path.join(self.build_temp, 'libtaichi_core.so'),
                            os.path.join(core_dir, 'taichi_core.so'))
            elif get_os_name() == 'osx':
                shutil.copy(
                    os.path.join(self.build_temp, 'libtaichi_core.dylib'),
                    os.path.join(core_dir, 'taichi_core.so'))
                moltenvk_path = os.path.join(self.build_temp,
                                             'libMoltenVK.dylib')
                if os.path.exists(moltenvk_path):
                    shutil.copy(moltenvk_path,
                                os.path.join(runtime_dir, 'libMoltenVK.dylib'))
            else:
                shutil.copy('runtimes/Release/taichi_core.dll',
                            os.path.join(core_dir, 'taichi_core.pyd'))

            if get_os_name() != 'osx':
                libdevice_path = 'external/cuda_libdevice/slim_libdevice.10.bc'
                print("copying libdevice:", libdevice_path)
                assert os.path.exists(libdevice_path)
                shutil.copy(libdevice_path,
                            os.path.join(runtime_dir, 'slim_libdevice.10.bc'))

            llvm_runtime_dir = 'taichi/runtime/llvm'
            for f in os.listdir(llvm_runtime_dir):
                if f.startswith('runtime_') and f.endswith('.bc'):
                    print(f"Fetching runtime file {f} to {target} folder")
                    shutil.copy(os.path.join(llvm_runtime_dir, f), runtime_dir)


class Clean(clean):
    def run(self):
        super().run()
        self.build_temp = os.path.join(root_dir, 'build')
        if os.path.exists(self.build_temp):
            remove_tree(self.build_temp, dry_run=self.dry_run)
        generated_folders = ('bin', 'dist', 'python/taichi/assets',
                             'python/taichi/lib/runtime',
                             'python/taichi.egg-info')
        for d in generated_folders:
            if os.path.exists(d):
                remove_tree(d, dry_run=self.dry_run)
        generated_files = [
            'taichi/common/commit_hash.h', 'taichi/common/version.h'
        ]
        generated_files += glob.glob('taichi/runtime/llvm/runtime_*.bc')
        generated_files += glob.glob('taichi/runtime/llvm/runtime_*.ll')
        for f in generated_files:
            if os.path.exists(f):
                print(f'removing generated file {f}')
                if not self.dry_run:
                    os.remove(f)


setup(name=project_name,
      packages=packages,
      package_dir={"": package_dir},
      version=version,
      description='The Taichi Programming Language',
      author='Taichi developers',
      author_email='yuanmhu@gmail.com',
      url='https://github.com/taichi-dev/taichi',
      python_requires=">=3.6,<3.10",
      install_requires=[
          'numpy',
          'sourceinspect>=0.0.4',
          'colorama',
          'astor',
      ],
      data_files=[('lib/runtime', data_files)],
      keywords=['graphics', 'simulation'],
      license='MIT',
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'ti=taichi.main:main',
          ],
      },
      classifiers=classifiers,
      ext_modules=[CMakeExtension('taichi_core')],
      cmdclass=dict(egg_info=EggInfo,
                    build_py=BuildPy,
                    build_ext=CMakeBuild,
                    clean=Clean),
      has_ext_modules=lambda: True)
