---
sidebar_position: 3
---

# Developer installation

:::note
End users should use the pip packages instead of building from source.
:::

This section documents how to configure the Taichi devolopment environment and build Taichi from source for the compiler developers. The installation instructions might vary between different operationg systems. We also provide a Dockerfile which may help setup a containerized Taichi development environment with CUDA support based on the Ubuntu base docker image.

### Installing Dependencies
1. Make sure you are using Python 3.6/3.7/3.8/3.9.
  - If you are on an Apple M1 machine, you might want to install `conda` from [Miniforge](https://github.com/conda-forge/miniforge/#download).

2.  Make sure you have `clang` with version \>= 7 on Linux or download clang-10 on Windows:

  - On OSX: you don’t need to do anything.
  - On Ubuntu, execute `sudo apt install libtinfo-dev clang-8`.
  - On Arch Linux, execute `sudo pacman -S clang`. (This is
    `clang-10`).
  - On other Linux distributions, please search [this
    site](https://pkgs.org) for clang version \>= 7.
  - On Windows please download [clang-10](https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/clang-10.0.0-win.zip). Make sure you add the `bin` folder containing `clang.exe` to the `PATH` environment variable.


:::note
Note that on Linux, `clang` is the **only** supported compiler for compiling the Taichi package.
:::


3. Make sure you have LLVM 10.0.0. Note that Taichi uses a **customized
  LLVM** so the pre-built binaries from the LLVM official website or
  other sources probably won't work. Here we provide LLVM binaries
  customized for Taichi, which may or may not work depending on your
  system environment:
  - [LLVM 10.0.0 for Linux](https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-linux.zip)
  - [LLVM 10.0.0 for macOS](https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-macos.zip)
  - [LLVM 10.0.0 for Windows MSVC 2019](https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-msvc2019.zip)

  :::note
  If you use the pre-built LLVM for Taichi, please add `$LLVM_FOLDER/bin` to `PATH`, e.g. `export PATH=<path_to_llvm_folder>/bin:$PATH` on Linux.
  :::

- If the downloaded LLVM does not work, please build from source:
  - For Linux & Mac OSX:

    ```bash
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/llvm-10.0.0.src.tar.xz
    tar xvJf llvm-10.0.0.src.tar.xz
    cd llvm-10.0.0.src
    mkdir build
    cd build
    cmake .. -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_TERMINFO=OFF
    # If you are building on Apple M1, use -DLLVM_TARGETS_TO_BUILD="AArch64".
    # If you are building on NVIDIA Jetson TX2, use -DLLVM_TARGETS_TO_BUILD="ARM;NVPTX"
    # If you are building for a PyPI release, add -DLLVM_ENABLE_Z3_SOLVER=OFF to reduce the library dependency.
    make -j 8
    sudo make install
    # Check your LLVM installation
    llvm-config --version  # You should get 10.0.0
    ```

  - For Windows:

    ```bash
    # For Windows
    # LLVM 10.0.0 + MSVC 2019
    cmake .. -G"Visual Studio 16 2019" -A x64 -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF   -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64   -DLLVM_BUILD_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=installed
    ```

    - Then open `LLVM.sln` and use Visual Studio 2017+ to build.
    - Please make sure you are using the `Release` configuration.
      After building the `INSTALL` project (under folder
      `CMakePredefinedTargets` in the Solution Explorer window).
    - If you use MSVC 2019, **make sure you use C++17** for the
      `INSTALL` project.
    - After the build is complete, find your LLVM binaries and
      headers in `build/installed`.
    - Please add `build/installed/bin` to `PATH`. Later, when you
    build Taichi please use
    `cmake -DLLVM_DIR=<path_to_build>/build/installed/lib/cmake/llvm`.


### Setting up CUDA (optional)

:::note
To build with NVIDIA GPU support, CUDA 10.0+ is needed. This
installation guide works for Ubuntu 16.04+.
:::

If you don't have CUDA, go to [this
website](https://developer.nvidia.com/cuda-downloads) and download the
installer.

- To check if CUDA is installed, run `nvcc --version` or
  `cat /usr/local/cuda/version.txt`.
- On **Ubuntu** we recommend choosing `deb (local)` as **Installer
  Type**.
- On **Arch Linux**, you can easily install CUDA via `pacman -S cuda`
  without downloading the installer manually.



### Setting up Vulkan (optional)

If you wish to build taichi with Vulkan. You will need to install the Vulkan SDK.
Please visit [this website](https://vulkan.lunarg.com/sdk/home) and follow the instructions for your OS.
If you're working on Windows, please also set the environment variable `VULKAN_SDK` to `C:/VulkanSDK/${YOUR_VULKAN_VERSION}`.
(as an example, for vulkan 1.2.189.0, set `VULKAN_SDK` to `C:/VulkanSDK/1.2.189.0`).
Finally, please add an environment variable `TAICHI_CMAKE_ARGS` with the value `-DTI_WITH_VULKAN:BOOL=ON` to enable building vulkan with taichi.

### Setting up Taichi for development

1. Clone the Taichi repo **recursively**, and build:

  ```bash
  git clone --recursive https://github.com/taichi-dev/taichi
  cd taichi
  python3 -m pip install --user -r requirements_dev.txt
  # export CXX=/path/to/clang  # Uncomment if clang is not system default compiler.
  python3 setup.py develop --user  # Optionally add DEBUG=1 to keep debug information.
  ```
  - We use MSBUILD.exe to build the generated project on Windows. Please note that Windows
    could have multiple instances of MSBUILD.exe shipped with different
    products. Please make sure you add the path for MSBUILD.exe within your
    MSVS directory and make it a higher priority (for instance than the one
    shipped with .NET).

2. Check out `examples` for runnable examples. Run them with commands
  like `python3 examples/simulation/mpm128.py`.

3. Execute `python3 -m taichi test` to run all the tests. It may take
  up to 5 minutes to run all tests.

## Conda

To avoid directly installing Taichi's dependencies into your existing
Python environment, we have provided a pre-defined `conda` environment.
You can find the instructions [here](https://github.com/taichi-dev/taichi/blob/master/conda/README.md).

Note that this step only helps you setup the development environment,
you would still need to run `python3 setup.py develop` to re-build
Taichi.

## Docker

For those who prefer to use Docker, we also provide a Dockerfile which
helps setup the Taichi development environment with CUDA support based
on Ubuntu docker image.

:::note
In order to follow the instructions in this section, please make sure
you have the [Docker DeskTop (or Engine for
Linux)](https://www.docker.com/products/docker-desktop) installed and
set up properly.
:::

### Build the Docker Image

From within the root directory of the taichi Git repository, execute
`docker build -t taichi:latest .` to build a Docker image based off the
local master branch tagged with _latest_. Since this builds the image
from source, please expect up to 40 mins build time if you don't have
cached Docker image layers.

:::note

In order to save the time on building Docker images, you could always
visit our [Docker Hub
repository](https://hub.docker.com/r/taichidev/taichi) and pull the
versions of pre-built images you would like to use. Currently the builds
are triggered per taichi Github release.

For example, to pull a image built from release v0.6.17, run
`docker pull taichidev/taichi:v0.6.17`
:::

:::caution

The nature of Docker container determines that no changes to the file
system on the container could be preserved once you exit from the
container. If you want to use Docker as a persistent development
environment, we recommend you [mount the taichi Git repository to the
container as a volume](https://docs.docker.com/storage/volumes/) and set
the Python path to the mounted directory.
:::

### Use Docker Image on macOS (cpu only)

1.  Make sure `XQuartz` and `socat` are installed:

```bash
brew cask install xquartz
brew install socat
```

2.  Temporally disable the xhost access-control: `xhost +`
3.  Start the Docker container with
    `docker run -it -e DISPLAY=$(ipconfig getifaddr en0):0 taichidev/taichi:v0.6.17`
4.  Do whatever you want within the container, e.g. you could run tests
    or an example, try: `ti test` or `ti example mpm88`
5.  Exit from the container with `exit` or `ctrl+D`
6.  \[To keep your xhost safe\] Re-enable the xhost access-control:
    `xhost -`

### Use Docker Image on Ubuntu (with CUDA support)

1.  Make sure your host machine has CUDA properly installed and
    configured. Usually you could verify it by running `nvidia-smi`
2.  Make sure [NVIDIA Container
    Toolkit](https://github.com/NVIDIA/nvidia-docker) is properly
    installed:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

3.  Make sure `xorg` is installed: `sudo apt-get install xorg`
4.  Temporally disable the xhost access-control: `xhost +`
5.  Start the Docker container with
    `sudo docker run -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix taichidev/taichi:v0.6.17`
6.  Do whatever you want within the container, e.g. you could run tests
    or an example, try: `ti test` or `ti example mpm88`
7.  Exit from the container with `exit` or `ctrl+D`
8.  **[To keep your xhost safe]** Re-enable the xhost access-control:
    `xhost -`


## Troubleshooting Developer Installation

- If `python3 setup.py develop`(or `python3 setup.py install`) gives `permission denied` error, it means you're
  installing into system python without write permission. You can work around this by:
  - `python3 setup.py develop --user` or `python3 setup.py install --user`
  - Install conda and use python from conda enviroments.

- If `make` fails to compile and reports
  `fatal error: 'spdlog/XXX.h' file not found`, please try runing
  `git submodule update --init --recursive --depth=1`.

- If the build succeeded but running any Taichi code results in errors
  like

  ```
  Bitcode file (/tmp/taichi-tero94pl/runtime//runtime_x64.bc) not found
  ```

  please double check `clang` is in your `PATH`:

  ```bash
  clang --version
  # version should be >= 7
  ```

  and our **Taichi configured** `llvm-as`:

  ```bash
  llvm-as --version
  # version should be >= 8
  which llvm-as
  # should be /usr/local/bin/llvm-as or /opt/XXX/bin/llvm-as, which is our configured installation
  ```

  If not, please install `clang` and **build LLVM from source** with
  instructions above in [dev_install](#installing-dependencies-1),
  then add their path to environment variable `PATH`.

- If you don't have `wget` on OSX, try installing [homebrew](https://brew.sh/) and then run `brew install wget`.

- If you get a new Apple machine, you might need to run `xcode-select --install` first.

- If you installed `conda` but `which python` still points to the system `python` location, run the following commands to enable it:

  ```
  source <path_to_conda>/bin/activate
  conda init
  ```

- If you encounter other issues, feel free to report (please include the details) by [opening an
  issue on
  GitHub](https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md).
  We are willing to help!

- See also [Installation Troubleshooting](../misc/install.md) for issues
  that may share with end-user installation.
