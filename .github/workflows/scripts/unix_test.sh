TAICHI_REPO_DIR=`pwd`
TI_LIB_DIR=`python3 -c "import taichi;print(taichi.__path__[0])" | tail -1`
TI_LIB_DIR="$TI_LIB_DIR/lib" ./build/taichi_cpp_tests
export PATH=$TAICHI_REPO_DIR/taichi-llvm/bin/:$PATH
hash -r
python3 examples/algorithm/laplace.py
ti diagnose
ti changelog
[ -z $GPU_TEST ] && ti test -vr2 -t2

[ -z $GPU_TEST ] || ti test -vr2 -t2 -k "not ndarray"
[ -z $GPU_TEST ] || ti test -vr2 -t2 -k "ndarray"
