pipeline {
    agent any
    environment {
        PATH = "/usr/local/clang-7.0.1/bin:/usr/local/cuda/bin/:$PATH"
        LD_LIBRARY_PATH = "/usr/local/clang-7.0.1/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        CC = "clang-7"
        CXX = "clang++"
    }
    stages{
        stage('Build') {
            parallel {
                stage('cuda10.0-python3.6') {
                    agent {
                        node {
                            label "cuda10.0 && python3.6"
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
                            label "cuda10.0 && python3.7"
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
                            label "cuda10.1 && python3.6"
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
                            label "cuda10.1 && python3.7"
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
    echo $WORKSPACE
    export TAICHI_REPO_DIR=$WORKSPACE/
    export PYTHONPATH=$TAICHI_REPO_DIR/python
    export PATH=$WORKSPACE/bin/:$PATH
    nvidia-smi
    $CC --version
    $CXX --version
    echo $TAICHI_REPO_DIR
    cd $TAICHI_REPO_DIR
    [ -e build ] && rm -rf build
    mkdir build && cd build
    cmake .. -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE -DCUDA_VERSION=$CUDA_VERSION
    make -j 15
    $CC --version
    $CXX --version
    echo $TAICHI_REPO_DIR
    echo $TAICHI_REPO_DIR
    echo $PYTHONPATH
    '''
}
