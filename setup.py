import glob
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys

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
TI_VERSION_MINOR = 7
TI_VERSION_PATCH = 28
version = f'{TI_VERSION_MAJOR}.{TI_VERSION_MINOR}.{TI_VERSION_PATCH}'

data_files = glob.glob('python/lib/*')
print(data_files)
packages = find_packages('python')
print(packages)

# Our python package root dir is python/
package_dir = 'python'


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
    assert False, "Unknown platform name %s" % name


def remove_tmp(taichi_dir):
    shutil.rmtree(os.path.join(taichi_dir, 'assets'), ignore_errors=True)
    shutil.rmtree(os.path.join(taichi_dir, 'examples'), ignore_errors=True)
    shutil.rmtree(os.path.join(taichi_dir, 'tests'), ignore_errors=True)


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class EggInfo(egg_info):
    def run(self):
        taichi_dir = os.path.join(package_dir, 'taichi')
        remove_tmp(taichi_dir)

        shutil.copytree('tests/python', os.path.join(taichi_dir, 'tests'))
        shutil.copytree('examples', os.path.join(taichi_dir, 'examples'))
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
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
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
        taichi_lib_dir = 'taichi/lib'
        for target in (
                os.path.join(package_dir, taichi_lib_dir),
                os.path.join(self.build_lib, taichi_lib_dir),
        ):
            shutil.rmtree(target, ignore_errors=True)
            os.makedirs(target)
            if get_os_name() == 'linux':
                shutil.copy(os.path.join(self.build_temp, 'libtaichi_core.so'),
                            os.path.join(target, 'taichi_core.so'))
            elif get_os_name() == 'osx':
                shutil.copy(
                    os.path.join(self.build_temp, 'libtaichi_core.dylib'),
                    os.path.join(target, 'taichi_core.so'))
            else:
                shutil.copy('runtimes/Release/taichi_core.dll',
                            os.path.join(target, 'taichi_core.pyd'))

            if get_os_name() != 'osx':
                libdevice_path = 'external/cuda_libdevice/slim_libdevice.10.bc'
                print("copying libdevice:", libdevice_path)
                assert os.path.exists(libdevice_path)
                shutil.copy(libdevice_path,
                            os.path.join(target, 'slim_libdevice.10.bc'))

            llvm_runtime_dir = 'taichi/runtime/llvm'
            for f in os.listdir(llvm_runtime_dir):
                if f.startswith('runtime_') and f.endswith('.bc'):
                    print(f"Fetching runtime file {f} to {target} folder")
                    shutil.copy(os.path.join(llvm_runtime_dir, f), target)


setup(name=project_name,
      packages=packages,
      package_dir={"": package_dir},
      version=version,
      description='The Taichi Programming Language',
      author='Taichi developers',
      author_email='yuanmhu@gmail.com',
      url='https://github.com/taichi-dev/taichi',
      install_requires=[
          'numpy',
          'pybind11>=2.5.0',
          'sourceinspect>=0.0.4',
          'colorama',
          'astor',
      ],
      data_files=[('lib', data_files)],
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
      cmdclass=dict(egg_info=EggInfo, build_py=BuildPy, build_ext=CMakeBuild),
      has_ext_modules=lambda: True)
