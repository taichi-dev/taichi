#!/bin/bash

# gcc : CXX="g++  -finline-limit=10000 -ftemplate-depth-2000 --param max-inline-recursive-depth=2000"
# icc : CXX="icpc -fast -no-inline-max-size -fno-exceptions"
CXX=${CXX-g++  -finline-limit=10000 -ftemplate-depth-2000 --param max-inline-recursive-depth=2000} # default value

for ((i=1; i<16; ++i)); do
    echo "Matrix size: $i x $i :"
    $CXX -O3 -I.. -DNDEBUG  benchmark.cpp -DMATSIZE=$i -DEIGEN_UNROLLING_LIMIT=400 -o benchmark && time ./benchmark >/dev/null
    $CXX -O3 -I.. -DNDEBUG -finline-limit=10000 benchmark.cpp -DMATSIZE=$i -DEIGEN_DONT_USE_UNROLLED_LOOPS=1 -o benchmark && time ./benchmark >/dev/null
    echo " "
done
