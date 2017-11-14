wget https://github.com/embree/embree/releases/download/v2.14.0/embree-2.14.0.x86_64.macosx.tar.gz
tar -xzf embree-2.14.0.x86_64.macosx.tar.gz
source embree-2.14.0.x86_64.macosx/embree-vars.sh
cp embree-2.14.0.x86_64.macosx/lib/*.dylib .
cmake .. -DEMBREE_INCLUDE_PATH:PATH=embree-2.14.0.x86_64.macosx/include -DEMBREE_LIBRARY=libembree.2.dylib -DTC_DISABLE_SIMD:BOOL=true
make -j4
