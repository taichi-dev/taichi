# Optional environment variables supported by setup.py:
#   {DEBUG, RELWITHDEBINFO, MINSIZEREL}
#     build the C++ taichi_core extension with various build types.
#
#   TAICHI_CMAKE_ARGS
#     extra cmake args for C++ taichi_core extension.

import glob
import multiprocessing
import os
import shutil
import sys
from distutils.command.clean import clean
from distutils.dir_util import remove_tree

from setuptools import find_packages
from skbuild import setup
from skbuild.command.egg_info import egg_info

root_dir = os.path.dirname(os.path.abspath(__file__))

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Software Development :: Compilers',
    'Topic :: Multimedia :: Graphics',
    'Topic :: Games/Entertainment :: Simulation',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]


def get_version():
    if os.getenv("RELEASE_VERSION"):
        version = os.environ["RELEASE_VERSION"]
    else:
        version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
        with open(version_file, 'r') as f:
            version = f.read().strip()
    return version.lstrip("v")


project_name = os.getenv('PROJECT_NAME', 'taichi')
version = get_version()
TI_VERSION_MAJOR, TI_VERSION_MINOR, TI_VERSION_PATCH = version.split('.')

data_files = glob.glob('python/_lib/runtime/*')
print(data_files)
packages = find_packages('python')
print(packages)

# Our python package root dir is python/
package_dir = 'python'


def remove_tmp(taichi_dir):
    shutil.rmtree(os.path.join(taichi_dir, 'assets'), ignore_errors=True)


class EggInfo(egg_info):
    def finalize_options(self, *args, **kwargs):
        if '' not in self.distribution.package_dir:
            # Issue#4975: skbuild loses the root package dir
            self.distribution.package_dir[''] = package_dir
        return super().finalize_options(*args, **kwargs)


def copy_assets():
    taichi_dir = os.path.join(package_dir, 'taichi')
    remove_tmp(taichi_dir)

    shutil.copytree('external/assets', os.path.join(taichi_dir, 'assets'))


class Clean(clean):
    def run(self):
        super().run()
        self.build_temp = os.path.join(root_dir, '_skbuild')
        if os.path.exists(self.build_temp):
            remove_tree(self.build_temp, dry_run=self.dry_run)
        generated_folders = ('bin', 'dist', 'python/taichi/assets',
                             'python/taichi/_lib/runtime', 'taichi.egg-info',
                             'python/taichi.egg-info', 'build')
        for d in generated_folders:
            if os.path.exists(d):
                remove_tree(d, dry_run=self.dry_run)
        generated_files = [
            'taichi/common/commit_hash.h', 'taichi/common/version.h'
        ]
        generated_files += glob.glob('taichi/runtime/llvm/runtime_*.bc')
        generated_files += glob.glob('python/taichi/_lib/core/*.so')
        generated_files += glob.glob('python/taichi/_lib/core/*.pyd')
        for f in generated_files:
            if os.path.exists(f):
                print(f'removing generated file {f}')
                if not self.dry_run:
                    os.remove(f)


def get_cmake_args():
    import shlex

    num_threads = os.getenv('BUILD_NUM_THREADS', multiprocessing.cpu_count())
    cmake_args = shlex.split(os.getenv('TAICHI_CMAKE_ARGS', '').strip())

    if (os.getenv('DEBUG', '0') in ('1', 'ON')):
        cfg = 'Debug'
    elif (os.getenv('RELWITHDEBINFO', '0') in ('1', 'ON')):
        cfg = 'RelWithDebInfo'
    elif (os.getenv('MINSIZEREL', '0') in ('1', 'ON')):
        cfg = 'MinSizeRel'
    else:
        cfg = None
    build_options = []
    if cfg:
        build_options.extend(['--build-type', cfg])
    if sys.platform == 'win32':
        build_options.extend(['-G', 'Ninja', '--skip-generator-test'])
    sys.argv[2:2] = build_options

    cmake_args += [
        f'-DTI_VERSION_MAJOR={TI_VERSION_MAJOR}',
        f'-DTI_VERSION_MINOR={TI_VERSION_MINOR}',
        f'-DTI_VERSION_PATCH={TI_VERSION_PATCH}',
    ]
    emscriptened = os.getenv('TI_EMSCRIPTENED', '0') in ('1', 'ON')
    if emscriptened:
        cmake_args += ['-DTI_EMSCRIPTENED=ON']

    if sys.platform != 'win32':
        os.environ['SKBUILD_BUILD_OPTIONS'] = f'-j{num_threads}'
    return cmake_args


def exclude_paths(manifest_files):
    return [
        f for f in manifest_files
        if f.endswith(('.so', 'pyd',
                       '.bc')) or os.path.basename(f) == 'libMoltenVK.dylib'
    ]


copy_assets()
setup(name=project_name,
      packages=packages,
      package_dir={"": package_dir},
      version=version,
      description='The Taichi Programming Language',
      author='Taichi developers',
      author_email='yuanmhu@gmail.com',
      url='https://github.com/taichi-dev/taichi',
      python_requires=">=3.6,<3.11",
      install_requires=[
          'numpy', 'sourceinspect>=0.0.4', 'colorama',
          'astunparse;python_version<"3.9"'
      ],
      data_files=[(os.path.join('_lib', 'runtime'), data_files)],
      keywords=['graphics', 'simulation'],
      license=
      'Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'ti=taichi._main:main',
          ],
      },
      classifiers=classifiers,
      cmake_args=get_cmake_args(),
      cmake_process_manifest_hook=exclude_paths,
      cmdclass={
          'egg_info': EggInfo,
          'clean': Clean
      },
      has_ext_modules=lambda: True)
