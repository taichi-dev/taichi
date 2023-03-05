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
  - [Troubleshooting and debugging](#troubleshooting-and-debugging)
  - [Frequently asked questions](#frequently-asked-questions)

:::note

Installation instructions vary depending on which operating system (OS) you are using. Choose the right OS or platform before you proceed.

:::

## Prerequisites

````mdx-code-block
<Tabs
  defaultValue="unix"
  values={[
    {label: 'Linux/Mac', value: 'unix'},
    {label: 'Windows', value: 'windows'}
  ]}>

<TabItem value="unix">

| Category                     | Prerequisites                                                                                                                                                                            |
|:----------------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OS                           | macOS / Ubuntu / Arch Linux / Other Linux distributions                                                                                                       |
| Python                       | 3.7/3.8/3.9/3.10 We recommend installing Python from [Miniforge](https://github.com/conda-forge/miniforge/#download) conda if you are on a MacBook with M1 chip. |
| Clang++                      | Clang++ &gt;8                            |
| LLVM                         | 15.0.4 (Taichi customized version)                                                                                                                                                       |
| Command line tools for Xcode | For macOS users only: `xcode-select --install `                                                                                                                                          |

</TabItem>

<TabItem value="windows">

| Category      | Prerequisites                                                                                                                                                                            |
|:-------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OS            | Windows 7/8/10/11                                                                                                       |
| Python        | 3.7/3.8/3.9/3.10  |
| Clang++       | Clang++ &gt;8                           |
| LLVM          | 15.0.4 (Taichi customized version)                                                                                                                                                       |
| Visual Studio | Visual Studio 2019/2022 with "Desktop Development with C++" component. If you want to use Clang++ as the compiler, also install "C++ Clang Compiler for Windows" component  |

</TabItem>

</Tabs>
````
### Install Clang

<blockquote>

Taichi supports building from source with any clang compiler greater than version 8.0. Install one from your favorite package tool for the operating system. For example, you can use `apt` on Ubuntu, `brew` on macOS, and so on.
</blockquote>


### Install LLVM

#### Install pre-built, customized LLVM binaries

We provide pre-built, customized LLVM 15 binaries.

1. Download and install customized binaries from the following list per your system environment:

````mdx-code-block
<Tabs
  defaultValue="llvm_linux"
  values={[
    {label: 'LLVM 15 for Linux', value: 'llvm_linux'},
    {label: 'LLVM 15 for macOS (without M1 chip)', value: 'llvm_macos_sans_m1'},
    {label: 'LLVM 15 for macOS (with M1 chip)', value: 'llvm_macos_m1'},
    {label: 'LLVM 15 for Windows', value: 'llvm_windows'},
  ]}>

<TabItem value="llvm_linux">
    <a href="https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip">LLVM 15.0.0 for Linux</a>
</TabItem>
<TabItem value="llvm_macos_sans_m1">
    <a href="https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/llvm-15-mac10.15.zip">LLVM 15.0.0 for macOS (without M1 chip)</a>
</TabItem>
<TabItem value="llvm_macos_m1">
    <a href="https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-m1.zip">LLVM 15.0.0 for macOS (with M1 chip)</a>
</TabItem>
<TabItem value="llvm_windows">
    <a href="https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15.0.0-msvc2019.zip">LLVM 15.0.0 for Windows MSVC 2019</a>
    <a href="https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15.0.3-msvc2022.zip">LLVM 15.0.0 for Windows MSVC 2022</a>
</TabItem>
</Tabs>

````

2. Configure environment variable:

````mdx-code-block
<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux & macOS', value: 'linux'},
    {label: 'Windows', value: 'windows'},
  ]}>

<TabItem value="linux">

1. Set `LLVM_DIR` environment variable:
   ```
   echo "export LLVM_DIR=<PATH_TO_LLVM_FOLDER>" >>  ~/.bashrc
   ```
2. Update your path for the remainder of the session:

   ```shell
   source ~/.bashrc
   ```

</TabItem>

<TabItem value="windows">

Add an environment variable `LLVM_DIR` with value `<Path to the extracted LLVM binary>`

</TabItem>

</Tabs>
````

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

<details>
<summary><font color="#006284">Install Vulkan</font></summary>

You must install the Vulkan SDK in order to debug Taichi's Vulkan backend. To proceed:

````mdx-code-block
<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux', value: 'linux'},
    {label: 'Windows', value: 'windows'},
]}>

<TabItem value="linux">

1. Go to [Vulkan's SDK download page](https://vulkan.lunarg.com/sdk/home) and follow the instructions for your OS.
2. Check if environment variables `VULKAN_SDK`, `PATH`, `LD_LIBRARY_PATH`, and `VK_LAYER_PATH` are updated.

  > The SDK for Ubuntu provides a `setup-env.sh` for updating these variables.

3. Ensure that you have a Vulkan driver from a GPU vendor properly installed.

  > On Ubuntu, check if a JSON file with a name corresponding to your GPU vendor is in: `/etc/vulkan/icd.d/` or `/usr/share/vulkan/icd.d/`.

4. Check if the SDK is properly installed: `vulkaninfo`.

5. If the SDK is properly installed, add an environment variable `TAICHI_CMAKE_ARGS` with the value `-DTI_WITH_VULKAN:BOOL=ON` to enable the Vulkan backend: (Otherwise Vulkan backend is disabled by default when compiling from source.)

  ```shell
  export TAICHI_CMAKE_ARGS="$TAICHI_CMAKE_ARGS -DTI_WITH_VULKAN:BOOL=ON"
  ```

</TabItem>

<TabItem value="windows">

1. Go to [Vulkan's SDK download page](https://vulkan.lunarg.com/sdk/home) and follow the instructions for your OS.
2. Set the environment variable `VULKAN_SDK` to `C:/VulkanSDK/${YOUR_VULKAN_VERSION}`.
3. If the SDK is properly installed, add an environment variable `TAICHI_CMAKE_ARGS` with the value `-DTI_WITH_VULKAN:BOOL=ON` to enable the Vulkan backend:

  ```shell
  $env:TAICHI_CMAKE_ARGS += " -DTI_WITH_VULKAN:BOOL=ON"
  ```

</TabItem>

</Tabs>
````
</details>

## Build Taichi from source

````mdx-code-block
<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux & macOS', value: 'linux'},
    {label: 'Windows', value: 'windows'},
  ]}>

<TabItem value="linux">

1. Clone the Taichi repo *recursively* and build[^1]:

  ```shell
  git clone --recursive https://github.com/taichi-dev/taichi

  cd taichi

  python3 -m pip install --user -r requirements_dev.txt

  # export CXX=/path/to/clang++  # Uncomment if clang++ is not default compiler of the system. Note that clang is not acceptable due to requirements of some submodules.

  # export DEBUG=1 #Uncomment it if you wish to keep debug information.

  python3 setup.py develop --user
  ```

2. Try out some of the demos in the **examples/** folder to see if Taichi is properly installed. For example:

  ```shell
  python3 python/taichi/examples/simulation/mpm128.py
  ```

:::note

[^1]Although the two commands work similarly, `python setup.py develop` is recommended for you as a developer and `python setup.py install`more for end users. The difference is:

- The `develop` command does not actually install anything but only symbolically links the source code to the deployment directory.
- The `install` command deep copies the source code so that end users need to rerun the command every time they modify the source code.

The `develop` command serves the developers' needs better because edits to the Python files take effect immediately without the need to rerun the command. A rerun is needed only if you have modified the project's C extension or compiled files. See the [Development Mode](https://setuptools.pypa.io/en/stable/userguide/development_mode.html) for more information.

:::

</TabItem>

<TabItem value="windows">

1. Set-up the environment variable `TAICHI_CMAKE_ARGS` with value `-DCLANG_EXECUTABLE=<Path to Clang>/bin/clang.exe -DLLVM_AS_EXECUTABLE=<Path to LLVM 15>/bin/llvm-as.exe`
2. Open the "x64 Native Tools Command Prompt" for VS2019 or VS2022. Please make sure you opened the x64 version. (Or load the Visual Studio environment yourself)
3. Clone the Taichi repo *recursively* & install python dependencies


   ```shell
   git clone --recursive https://github.com/taichi-dev/taichi

   cd taichi

   python -m pip install --user -r requirements_dev.txt
   ```

4. Build taichi by using `python setup.py develop`

:::note

[^1]Although the two commands work similarly, `python setup.py develop` is recommended for you as a developer and `python setup.py install`more for end users. The difference is:

- The `develop` command does not actually install anything but only symbolically links the source code to the deployment directory.
- The `install` command deep copies the source code so that end users need to rerun the command every time they modify the source code.

The `develop` command serves the developers' needs better because edits to the Python files take effect immediately without the need to rerun the command. A rerun is needed only if you have modified the project's C extension or compiled files. See the [Development Mode](https://setuptools.pypa.io/en/stable/userguide/development_mode.html) for more information.

:::

:::note

If you want to build Taichi with Clang or maybe utilize `ccache` to cache and speed-up builds, add the following to the end of environment variable `TAICHI_CMAKE_ARGS`: ` -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang`.

:::

</TabItem>

</Tabs>
````

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

`which python` still returns the system's Python location after Conda is installed.

**Workaround**

Run the following commands to activate Conda:

```shell
source <PATH_TO_CONDA>/bin/activate

conda init
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
