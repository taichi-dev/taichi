pipeline {
    agent any
    environment {
        PYPI_PWD = credentials("${PYPI_PWD}")
        PATH = "/opt/taichi-llvm-10.0.0/bin:/usr/local/cuda/bin/:$PATH"
        CC = "clang-10"
        CXX = "clang++-10"
        PYTHON_EXECUTABLE = "python3"
        // Local machine use 11.2, we pass a hack version to avoid build errors.
        HACK_CUDA_VERSION = "10.0"
    }
    stages{
        stage('Build') {
            parallel {
                stage('python3.6') {
                    agent {
                        node {
                            label "python36"
                            customWorkspace "taichi_py36"
                        }
                    }
                    environment {
                        CONDA_ENV = "py36"
                    }
                    steps{
                        build_taichi()
                    }
                }
                stage('python3.7') {
                    agent {
                        node {
                            label "python37"
                            customWorkspace "taichi_py37"
                        }
                    }
                    environment {
                        CONDA_ENV = "py37"
                    }
                    steps{
                        build_taichi()
                    }
                }
                stage('python3.8') {
                    agent {
                        node {
                            label "python38"
                            customWorkspace "taichi_py38"
                        }
                    }
                    environment {
                        CONDA_ENV = "py38"
                    }
                    steps{
                        build_taichi()
                    }
                }
                stage('python3.9') {
                    agent {
                        node {
                            label "python39"
                            customWorkspace "taichi_py39"
                        }
                    }
                    environment {
                        CONDA_ENV = "py39"
                    }
                    steps{
                        build_taichi()
                    }
                }
            }
        }
    }
}

void build_taichi() {
    sh "echo building"
    sh "echo $PATH"
    git 'https://github.com/taichi-dev/taichi.git'
    sh label: '', script: '''
    echo $PATH
    echo $CC
    echo $CXX
    $CC --version
    $CXX --version
    echo $WORKSPACE
    . "/home/buildbot/miniconda3/etc/profile.d/conda.sh"
    conda activate $CONDA_ENV
    $PYTHON_EXECUTABLE -m pip install --user setuptools astor pybind11 pylint sourceinspect
    $PYTHON_EXECUTABLE -m pip install --user pytest pytest-rerunfailures pytest-xdist yapf
    $PYTHON_EXECUTABLE -m pip install --user numpy GitPython coverage colorama autograd
    export TAICHI_REPO_DIR=$WORKSPACE
    echo $TAICHI_REPO_DIR
    export PYTHONPATH=$TAICHI_REPO_DIR/python
    export PATH=$WORKSPACE/bin/:$PATH
    nvidia-smi
    cd $TAICHI_REPO_DIR
    git submodule update --init --recursive
    [ -e build ] && rm -rf build
    mkdir build && cd build
    export CUDA_BIN_PATH=/usr/local/cuda-${HACK_CUDA_VERSION}
    cmake .. -DLLVM_DIR=/opt/taichi-llvm-10.0.0/lib/cmake/llvm \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DCUDA_VERSION=$HACK_CUDA_VERSION \
        -DTI_WITH_OPENGL=ON
    make -j 8
    ldd libtaichi_core.so
    objdump -T libtaichi_core.so| grep GLIBC
    cd ../python
    ti test -t 2
    $PYTHON_EXECUTABLE build.py upload ${TEST_OPTION}
    '''
}
