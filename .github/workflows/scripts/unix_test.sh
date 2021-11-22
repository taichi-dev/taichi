set -ex
TAICHI_REPO_DIR=`pwd`
python3 -m pip install -r requirements_test.txt
export PATH=$TAICHI_REPO_DIR/taichi-llvm/bin/:$PATH
## Only GPU machine uses system python.
[ -z $GPU_TEST ] || export PATH=$PATH:$HOME/.local/bin
hash -r
ti example laplace
ti diagnose
ti changelog
echo wanted archs: $TI_WANTED_ARCHS
[ -z $GPU_TEST ] && python tests/run_tests.py -vr2 -t2 -a "$TI_WANTED_ARCHS"

[ -z $GPU_TEST ] || python tests/run_tests.py -vr2 -t2 -k "not ndarray and not torch" -a "$TI_WANTED_ARCHS"
[ -z $GPU_TEST ] || python tests/run_tests.py -vr2 -t1 -k "ndarray or torch" -a "$TI_WANTED_ARCHS"
