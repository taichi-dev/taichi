pipeline {
    agent any
    environment {
        PYPI_PWD = credentials('PYPI_PWD')
        PATH = "/usr/local/clang-7.0.1/bin:/usr/local/cuda/bin/:$PATH"
        LD_LIBRARY_PATH = "/usr/local/clang-7.0.1/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        CC = "clang-7"
        CXX = "clang++-7"
        TI_WITH_CUDA = "True"
    }
    stages{
        stage('Build') {
            parallel {
                stage('cuda10.0-python3.6') {
                    agent {
                        node {
                            label "cuda10_0 && python3_6"
                            customWorkspace "taichi_cu100_py36"
                        }
                    }
                    environment {
                        PYTHON_EXECUTABLE = "python3.6"
                        CUDA_VERSION = "10.0"
                    }
                    steps{
                        build_taichi()
                    }
                }
                stage('cuda10.0-python3.7') {
                    agent {
                        node {
                            label "cuda10_0 && python3_7"
                            customWorkspace "taichi_cu100_py37"
                        }
                    }
                    environment {
                        PYTHON_EXECUTABLE = "python3.7"
                        CUDA_VERSION = "10.0"
                    }
                    steps{
                        build_taichi()
                    }
                }
                stage('cuda10.1-python3.6') {
                    agent {
                        node {
                            label "cuda10_1 && python3_6"
                            customWorkspace "taichi_cu101_py36"
                        }
                    }
                    environment {
                        PYTHON_EXECUTABLE = "python3.6"
                        CUDA_VERSION = "10.1"
                    }
                    steps{
                        build_taichi()
                    }
                }
                stage('cuda10.1-python3.7') {
                    agent {
                        node {
                            label "cuda10_1 && python3_7"
                            customWorkspace "taichi_cu101_py37"
                        }
                    }
                    environment {
                        PYTHON_EXECUTABLE = "python3.7"
                        CUDA_VERSION = "10.1"
                    }
                    steps{
                        build_taichi()
                    }
                }
                stage('cpu-python3.6') {
                    agent {
                        node {
                            label "python3_6"
                            customWorkspace "taichi_cpu_py36"
                        }
                    }
                    environment {
                        PYTHON_EXECUTABLE = "python3.6"
                        TI_WITH_CUDA = "False"
                    }
                    steps{
                        build_taichi()
                    }
                }
                stage('cpu-python3.7') {
                    agent {
                        node {
                            label "python3_7"
                            customWorkspace "taichi_cpu_py37"
                        }
                    }
                    environment {
                        PYTHON_EXECUTABLE = "python3.7"
                        TI_WITH_CUDA = "False"
                    }
                    steps{
                        build_taichi()
                    }
                }
            }
        }
        stage('Test') {
            steps {
                sh "echo Testing"
            }
        }
        stage('Release') {
            steps {
                sh "echo releasing"
            }
        }
    }
}

void build_taichi() {
    sh "echo building"
    sh "echo $PATH"
    git 'https://github.com/yuanming-hu/taichi.git'
    sh label: '', script: '''
    echo $PATH
    echo $CC
    echo $CXX
    $CC --version
    $CXX --version
    echo $WORKSPACE
    $PYTHON_EXECUTABLE -m pip install pytest autograd --user
    export TAICHI_REPO_DIR=$WORKSPACE/
    echo $TAICHI_REPO_DIR
    export PYTHONPATH=$TAICHI_REPO_DIR/python
    export PATH=$WORKSPACE/bin/:$PATH
    if [ "$TI_WITH_CUDA" != 'False' ]
    then
      nvidia-smi
    fi
    cd $TAICHI_REPO_DIR
    [ -e build ] && rm -rf build
    mkdir build && cd build
    export CUDA_BIN_PATH=/usr/local/cuda-${CUDA_VERSION}
    cmake .. -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE -DCUDA_VERSION=$CUDA_VERSION -DTLANG_WITH_CUDA:BOOL=$TI_WITH_CUDA
    make -j 15
    ldd libtaichi_core.so
    cd ../python
    ti test_python
    $PYTHON_EXECUTABLE build.py upload
    '''
}
