set -ex
TAICHI_REPO_DIR=`pwd`
python3 -m pip install -r requirements_test.txt
TI_LIB_DIR=`python3 -c "import taichi;print(taichi.__path__[0])" | tail -1`
[[ $RUN_CPP_TESTS == "ON" ]] && TI_LIB_DIR="$TI_LIB_DIR/lib" ./build/taichi_cpp_tests
export PATH=$TAICHI_REPO_DIR/taichi-llvm/bin/:$PATH
## Only GPU machine uses system python.
[ -z $GPU_TEST ] || export PATH=$PATH:$HOME/.local/bin
hash -r
python3 examples/algorithm/laplace.py
ti diagnose
ti changelog
[ -z $GPU_TEST ] && ti test -vr2 -t2

[ -z $GPU_TEST ] || ti test -vr2 -t2 -k "not ndarray and not torch"
[ -z $GPU_TEST ] || ti test -vr2 -t1 -k "ndarray or torch"
