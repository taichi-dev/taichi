export CXX="g++-6" CC="gcc-6"
cmake .. -DEMBREE_INCLUDE_PATH:PATH=/usr/include -DEMBREE_LIBRARY=/usr/lib64/libembree.so
make -j4
