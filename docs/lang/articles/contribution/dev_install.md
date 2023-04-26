---
sidebar_position: 2
---

# Developer Installation

## Target audience

Developers who are interested in the compiler, computer graphics, or high-performance computing, and would like to contribute new features or bug fixes to the [Taichi programming language](https://github.com/taichi-dev/taichi).

:::danger IMPORTANT

This installation guide is *NOT* intended for end users who only wish to do simulation or high performance numerical computation. We recommend that end users install Taichi via `pip install taichi`. There is no need for you to build Taichi from source.

See the [Get Started](https://docs.taichi-lang.org/) for more information on quickly setting up Taichi for end users.

:::


## Introduction

  This installation guide covers the following:

  - [Prerequisites for building Taichi from source](#prerequisites)
  - [Installing optional dependencies](#install-optional-dependencies)
  - [Building Taichi from source](#build-taichi-from-source)
  - [List of TAICHI_CMAKE_ARGS](#list-of-taichi_cmake_args)
  - [Usage and behavior of `build.py`](#usage-and-behavior-of-buildpy)
  - [Troubleshooting and debugging](#troubleshooting-and-debugging)
  - [Frequently asked questions](#frequently-asked-questions)

:::note

Installation instructions vary depending on which operating system (OS) you are using. Choose the right OS or platform before you proceed.

:::

:::note

With the release of Taichi v1.6.0, a comprehensive build environment preparation script (aka. `build.py` or `ti-build`) has been introduced. This script significantly simplifies the process of configuring a suitable build or development environment.

This guide will focus on the `build.py` approach. If you prefer to use the conventional method, you can refer to the previous Developer Installation document.

:::


## Prerequisites

````mdx-code-block
<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux', value: 'linux'},
    {label: 'Mac', value: 'mac'},
    {label: 'Windows', value: 'windows'}
  ]}>

<TabItem value="linux">

| Category                      | Prerequisites                                                |
| :---------------------------- | :----------------------------------------------------------- |
| Linux distribution            | Anything recent enough, e.g. Ubuntu 20.04                    |
| Python                        | 3.6+, with a usable pip(`python3-pip` package on Ubuntu)     |
| Clang++                       | Clang++ &gt;= 10, Clang++ 15 is recommended.                 |
| libstdc++-xx-dev              | Run `apt install libstdc++-10-dev`, or just install `g++`.   |

</TabItem>

<TabItem value="mac">

| Category                       | Prerequisites                            |
| :----------------------------- | ---------------------------------------- |
| macOS                          | macOS Big Sur or later                   |
| Python                         | 3.6+ (should be readily available)       |
| Command line tools for Xcode   | Run `xcode-select --install` to install  |

</TabItem>

<TabItem value="windows">

| Category        | Prerequisites                                                                   |
| :-------------- | ------------------------------------------------------------------------------- |
| Windows         | Windows 7/8/10/11                                                               |
| Python          | 3.6+                                                                            |
| Visual Studio   | Visual Studio 2022 (any edition) with "Desktop Development with C++" component. |

</TabItem>

</Tabs>
````

### Install Compiler

<blockquote>

Taichi supports building from source with Clang++ >= 10.0 and MSVC from VS2022.

For macOS developers, it is recommended to use AppleClang, which comes with the Command Line Tools for Xcode. You can install them by running `xcode-select --install`. Alternatively, you can also install Xcode.app from the Apple Store.

For Linux developers, it is recommended to install Clang using the package manager specific to your operating system. On Ubuntu 22.04, running `sudo apt install clang-15` should be sufficient. For older Ubuntu distributions to use a newer version of Clang, please follow the instructions on [official LLVM Debian/Ubuntu Nightly Packages](https://apt.llvm.org/).

For Windows developers, if none of the VS2022 editions are installed, `build.py` will automatically start a VS2022 BuildTools installer for you.

</blockquote>


### Install LLVM

#### Install pre-built, customized LLVM binaries

`build.py` will automatically download and setup a suitable version of pre-built LLVM binaries.

#### Alternatively, build LLVM from source

<details>
<summary><font color="#006284">Build LLVM 15.0.0 from source</font></summary>

We provide instructions here if you need to build LLVM 15.0.0 from source.

````mdx-code-block
<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux & macOS', value: 'linux'},
    {label: 'Windows', value: 'windows'},
  ]}>

<TabItem value="linux">

```shell
wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-15.0.5.tar.gz

tar zxvf llvmorg-15.0.5.tar.gz

cd llvm-project-llvmorg-15.0.5/llvm

mkdir build

cd build

cmake .. -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_TERMINFO=OFF

# If you are building on Apple M1, use -DLLVM_TARGETS_TO_BUILD="AArch64".

# If you are building on NVIDIA Jetson TX2, use -DLLVM_TARGETS_TO_BUILD="ARM;NVPTX"

# If you are building for a PyPI release, add -DLLVM_ENABLE_Z3_SOLVER=OFF to reduce the library dependency.

make -j 8

sudo make install

# Check your LLVM installation

llvm-config --version  # You should get 15.0.5
```

</TabItem>

<TabItem value="windows">

```shell
# For Windows

# LLVM 15.0.0 + MSVC 2019

cmake .. -G "Visual Studio 16 2019" -A x64 -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64 -DLLVM_BUILD_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=installed -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DCMAKE_CXX_STANDARD=17
cmake --build . --target=INSTALL --config=Release
```

1. Use Visual Studio 2017+ to build **LLVM.sln**.
2. Ensure that you use the **Release** configuration. After building the `INSTALL` project (under folder **CMakePredefinedTargets** in the Solution Explorer window).
3. If you use MSVC 2019+, ensure that you use **C++17** for the `INSTALL` project.
4. When the build completes, add an environment variable `LLVM_DIR` with value `<PATH_TO_BUILD>/build/installed/lib/cmake/llvm`.

</TabItem>

</Tabs>

To actually use the compiled LLVM binaries, replace the LLVM folder in the cache directory of `build.py` (open with `./build.py cache`) with your own version.
````

</details>

## Install optional dependencies

[CUDA](https://en.wikipedia.org/wiki/CUDA) is NVIDIA's answer to high-performance computing. Taichi has implemented a backend based on CUDA 10.0.0+. Vulkan is a next-generation, cross-platform API, open standard for 3D graphics and computing. Taichi has added a Vulkan backend as of v0.8.0.

This section provides instructions on installing these two optional dependencies.

<details>
<summary><font color="#006284">Install CUDA</font></summary>

This section works for you if you have a Nvidia GPU supporting CUDA. Note that the required CUDA version is 10.0+.

To install CUDA:

````mdx-code-block
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="ubuntu"
  values={[
    {label: 'Ubuntu', value: 'ubuntu'},
    {label: 'Arch Linux', value: 'arch'},
    {label: 'Windows', value: 'windows'},
  ]}>

<TabItem value="ubuntu">

1. Go to [the official site](https://developer.nvidia.com/cuda-downloads) to download the installer.
2. Choose **deb (local)** as **Installer Type**.
3. Check if CUDA is properly installed:

  ```
  nvidia-smi
  ```

</TabItem>

<TabItem value="arch">

1. `pacman -S cuda`
2. Check if CUDA is properly installed:

  ```
  nvidia-smi
  ```

</TabItem>

<TabItem value="windows">

1. Go to [the official site](https://developer.nvidia.com/cuda-downloads) and download the installer.
2. Choose **exe (local)** as **Installer Type**.
3. Check if CUDA is properly installed:

  ```
  nvidia-smi
  ```

</TabItem>

</Tabs>

````

</details>

Vulkan SDK is required to debug Taichi's Vulkan backend.
`build.py` will automatically download and setup a suitable version of Vulkan SDK.

On Windows, Vulkan SDK requires elevated privileges to install (the installer would set several machine scope environement variables).

<details>
<summary><font color="#006284">Ensure a working Vulkan SDK</font></summary>

````mdx-code-block
<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux', value: 'linux'},
    {label: 'Windows', value: 'windows'},
]}>

<TabItem value="linux">

1. Ensure that you have a Vulkan driver from a GPU vendor properly installed.

  > On Ubuntu, check if a JSON file with a name corresponding to your GPU vendor is in: `/etc/vulkan/icd.d/` or `/usr/share/vulkan/icd.d/`.

2. Add an environment variable `TAICHI_CMAKE_ARGS` with the value `-DTI_WITH_VULKAN:BOOL=ON` to enable the Vulkan backend: (Otherwise Vulkan backend is disabled by default when compiling from source, and `build.py` won't setup Vulkan SDK for you).

  ```shell
  export TAICHI_CMAKE_ARGS="$TAICHI_CMAKE_ARGS -DTI_WITH_VULKAN:BOOL=ON"
  ```

3. Check if the SDK is properly installed: Run `vulkaninfo` in the build shell:

    ```shell
    ./build.py --shell
    vulkaninfo
    ```

</TabItem>

<TabItem value="windows">

1. Add an environment variable `TAICHI_CMAKE_ARGS` with the value `-DTI_WITH_VULKAN:BOOL=ON` to enable the Vulkan backend: (Otherwise Vulkan backend is disabled by default when compiling from source, and `build.py` won't setup Vulkan SDK for you).

    ```pwsh
    $env:TAICHI_CMAKE_ARGS += " -DTI_WITH_VULKAN:BOOL=ON"
    ```

2. Check if the SDK is properly installed: Run `vulkaninfo` in the build shell:

    ```pwsh
    python ./build.py --shell
    vulkaninfo
    ```


</TabItem>

</Tabs>
````
</details>

## Build Taichi from source

1. Clone the Taichi repo *recursively* and build[^1]:

````mdx-code-block
<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux & macOS', value: 'linux'},
    {label: 'Windows', value: 'windows'},
  ]}>

<TabItem value="linux">

  ```shell
  git clone --recursive https://github.com/taichi-dev/taichi

  cd taichi

  # Customize with your own needs
  export TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON"

  # Uncomment if you want to use a different compiler
  # export CC=/path/to/clang
  # export CXX=/path/to/clang++

  # export DEBUG=1 # Uncomment it if you wish to keep debug information.

  # This would drop into a shell with complete build environment,
  ./build.py --shell

  # and then you could install Taichi in development mode
  python3 setup.py develop
  ```
</TabItem>

<TabItem value="windows">

  ```shell
  git clone --recursive https://github.com/taichi-dev/taichi

  cd taichi

  # Customize with your own needs
  $env:TAICHI_CMAKE_ARGS += " -DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON"

  # $env:DEBUG = 1 # Uncomment it if you wish to keep debug information.

  # This would drop into a shell with complete build environment,
  ./build.py --shell
  # and then you could install Taichi in development mode
  python3 setup.py develop
  ```

</TabItem>

</Tabs>
````

:::note

Alternatively, you could build a wheel file ready for install if you don't care about the convenience provided by `python develop install`:

```shell
./build.py
ls dist/*.whl
```

:::

2. Try out some of the demos in the **examples/** folder to see if Taichi is properly installed. For example:

  ```shell
  python3 python/taichi/examples/simulation/mpm128.py
  ```

:::note

[^1]Although the two commands work similarly, `./build.py --shell` and `python setup.py develop` is recommended for you as a developer and `./build.py` is more for end users. The difference is:

- The `python setup.py develop` command does not actually install anything but only symbolically links the source code to the deployment directory.
- The `./build.py` command builds a wheel suitable for shipping so that you need to rerun the command and install the wheel every time the source code is modified.

The `develop` command serves the developers' needs better because edits to the Python files take effect immediately without the need to rerun the command. A rerun is needed only if you have modified the project's C extension or compiled files. See the [Development Mode](https://setuptools.pypa.io/en/stable/userguide/development_mode.html) for more information.

:::

## List of TAICHI_CMAKE_ARGS

| Flag                         | Description                                                | Default |
| ---------------------------- | ---------------------------------------------------------- | ------- |
| BUILD_WITH_ADDRESS_SANITIZER | Build with clang address sanitizer                         | OFF     |
| TI_BUILD_EXAMPLES            | Build the C++ examples                                     | ON      |
| TI_BUILD_RHI_EXAMPLES        | Build the Unified Device API examples                      | OFF     |
| TI_BUILD_TESTS               | Build the C++ tests                                        | OFF     |
| TI_WITH_AMDGPU               | Build with the AMDGPU backend                              | OFF     |
| TI_WITH_BACKTRACE            | Use backward-cpp to print out C++ stack trace upon failure | OFF     |
| TI_WITH_CUDA                 | Build with the CUDA backend                                | ON      |
| TI_WITH_CUDA_TOOLKIT         | Build with the CUDA toolkit                                | OFF     |
| TI_WITH_C_API                | Build Taichi runtime C-API library                         | ON      |
| TI_WITH_DX11                 | Build with the DX11 backend                                | OFF     |
| TI_WITH_DX12                 | Build with the DX12 backend                                | OFF     |
| TI_WITH_GGUI                 | Build with GGUI                                            | OFF     |
| TI_WITH_GRAPHVIZ             | Generate dependency graphs between targets                 | OFF     |
| TI_WITH_LLVM                 | Build with LLVM backends                                   | ON      |
| TI_WITH_METAL                | Build with the Metal backend                               | ON      |
| TI_WITH_OPENGL               | Build with the OpenGL backend                              | ON      |
| TI_WITH_PYTHON               | Build with Python language binding                         | ON      |
| TI_WITH_STATIC_C_API         | Build static Taichi runtime C-API library                  | OFF     |
| TI_WITH_VULKAN               | Build with the Vulkan backend                              | OFF     |
| USE_LLD                      | Use lld (from llvm) linker                                 | OFF     |
| USE_MOLD                     | Use mold (A Modern Linker)                                 | OFF     |
| USE_STDCPP                   | Use -stdlib=libc++                                         | OFF     |


## Design goals, behaviors and usage of `build.py`

### Created to be dead simple

Setting up an appropriate development environment for an unfamiliar project can be quite challenging.
Therefore, `build.py` has been created to eliminate this friction. If you find any aspect of the environment configuration process to be
'too manual' or suffered to progress, it is considered a bug. Please report such issues on GitHub.

### Designed to be minimally intrusive

Nearly all the dependencies of `build.py` and Taichi are explicitly placed at the cache folder, which can be opened by:

```shell
./build.py cache
```

Or you can find it at:

| OS             | Cache Folder Location           |
| -------------- | ------------------------------- |
| Linux && macOS | `~/.cache/ti-build-cache`       |
| Windows        | `%LocalAppData%\ti-build-cache` |

A typical cache dir will contain sub folders below:

| Sub Folder       | Purpose                                                       | Code Responsible                                                                                                 |
| ---------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| bootstrap        | Contains Python packages used by `build.py` itself            | [bootstrap.py](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/ti_build/bootstrap.py) |
| deps             | Downloaded external dependencies, before extract/install      | [dep.py](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/ti_build/dep.py)             |
| llvm15           | Managed pre-built LLVM binaries                               | [llvm.py](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/ti_build/llvm.py)           |
| mambaforge       | Managed conda environment dedicated to build / develop Taichi | [python.py](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/ti_build/python.py)       |
| sccache          | Compile cache                                                 | [sccache.py](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/ti_build/sccache.py)     |
| vulkan-1.x.xxx.x | Vulkan SDK location                                           | [vulkan.py](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/ti_build/vulkan.py)       |

The whole cache folder can be safely removed.

`build.py` operates without the need for any third-party libraries to be installed, the requirements will be handled by its bootstrapping process.

:::note
On Debian/Ubuntu systems, `apt install python3-pip` is required.
:::

:::caution Behaviors considered intrusive

1. On Ubuntu systems, there's an attempt to install missing development libraries at [ospkg.py](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/ti_build/ospkg.py) by invoking `sudo apt install libxxxx-dev`
   if a terminal is detected. It can be skipped by telling `apt` not to install them.

2. Installing Vulkan SDK on Windows requires elevated privileges, and the installer will set several machine scoped environment variables (`VULKAN_SDK` and `VK_SDK_PATH`).

:::


### Choose your desired Python version, or use your own Python environment.

By default, `build.py` assumes that the same Python version used to invoke it will also be used for building Taichi.
`build.py` will then create an isolated Python environment and use it for all the subsequent Python related tasks.
To use a different version, please specify the desired version via `--python` option:

```shell
# Build a wheel
./build.py --python=3.10

# Or enter development shell
./build.py --python=3.10 --shell
```

If you prefer to manage Python environments yourself, you could specify `--python=native`, and `build.py` will not attempt to use a managed Python environment.

```shell
# Use your own conda
conda activate my-own-conda-env

# Build a wheel
./build.py --python=native

# Or enter development shell
./build.py --python=native --shell
```

## Troubleshooting and debugging

### Permission denied

**Description**

Gets a `permission denied` after `python3 setup.py develop` or `python3 setup.py install`.

**Root cause**

You were trying to install packages into the Python environment without write permission.

**Workaround**

1. `python3 setup.py develop --user` or `python3 setup.py install --user`.
2. Install Conda and use python from within the conda environment.

### `make` fails to compile

**Description**

`make` fails to compile and reports `fatal error: 'spdlog/XXX.h' file not found`.

**Root cause**

You did not use the `--recursive` flag when cloning the Taichi repository.

**Workaround**

Run `git submodule update --init --recursive --depth=1`.

### `which python` still returns the system's Python location

**Description**

`which python` still returns the system's Python location.

**Workaround**

Run the following commands to enter development shell:

```shell
./build.py --shell
```

## Frequently asked questions

### How can I get a fresh Taichi build?

1. Clean up cache from your previous builds:

  ```
  python3 setup.py clean
  ```

2. Uninstall the Taichi package from your Python environment:

- `python setup.py develop --uninstall`, if you build Taichi using `python setup.py develop`.
- `pip uninstall taichi`, if you build Taichi using `python setup.py install`.

### What if I don't have `wget` on my macOS?

1. Install [Homebrew](https://brew.sh/).
2. Use Homebrew to install `wget`:

  `brew install wget`

## Still have issues?

- See [Installation Troubleshooting](../faqs/install.md) for issues that may share with the end-user installation.

- If you encounter any issue that is not covered here, feel free to report it by [opening an issue on GitHub](https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md) and including the details. We are always there to help!
