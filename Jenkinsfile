pipeline {
    agent any
    environment {
        PYPI_PWD = credentials("${PYPI_PWD}")
        COMMIT_SHA = "${COMMIT_SHA}"
        PATH = "/opt/taichi-llvm-10.0.0/bin:/usr/local/cuda/bin/:$PATH"
        CC = "clang-10"
        CXX = "clang++-10"
        // Local machine uses version 11.2. However, we need to define
        // TI_CUDAVERSION, which eventually translates to the version number
        // of the slimmed CUDA libdevice bytecode. Currently this slimmed
        // version only covers 10. See:
        // https://github.com/taichi-dev/taichi/tree/master/external/cuda_libdevice
        // so we pass hack version to avoid build errors.
        HACK_CUDA_VERSION = "10.0"
    }
    stages{
        stage('Build and Test') {
            parallel {
                stage('python3.6') {
                    agent {
                        node {
                            label "python36"
                            customWorkspace "taichi_py36"
                        }
                    }
                    environment {
                        UBUNTU = "10.0-devel-ubuntu18.04"
                        PYTHON = "python3.6"
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
                        UBUNTU = "10.0-devel-ubuntu18.04"
                        PYTHON = "python3.7"
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
                        UBUNTU = "10.0-devel-ubuntu18.04"
                        PYTHON = "python3.8"
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
                        UBUNTU = "11.0-devel-ubuntu20.04"
                        PYTHON = "python3.9"
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
    cd ci
    docker build . --build-arg UBUNTU=${UBUNTU} --build-arg PYTHON=${PYTHON} --build-arg TEST_OPTION="${TEST_OPTION}" --build-arg PYPI_PWD=${PYPI_PWD} --build-arg COMMIT_SHA=${COMMIT_SHA}
    '''
}
