export PATH=`pwd`/taichi-llvm/bin/:$LLVM_PATH:$PATH
python3 -m pip uninstall taichi taichi-nightly -y
python3 -m pip install -r requirements_dev.txt
cd python
git fetch origin master
TAICHI_CMAKE_ARGS=$CI_SETUP_CMAKE_ARGS python3 build.py build
cd ..
export NUM_WHL=`ls dist/*.whl | wc -l`
if [ $NUM_WHL -ne 1 ]; then echo `ERROR: created more than 1 whl.` && exit 1; fi
python3 -m pip install dist/*.whl

# Show ELF info
TMP_DIR=__tmp
mkdir $TMP_DIR
unzip dist/*.whl -d $TMP_DIR
TACHICORE_PATH=$TMP_DIR/taichi/lib/taichi_core.so
ldd $TACHICORE_PATH
strings $TACHICORE_PATH | grep GLIBC
rm -fr $TMP_DIR
