#!/bin/bash
CXX=${CXX-g++} # default value unless caller has defined CXX
echo "Fixed size 3x3, column-major, -DNDEBUG"
$CXX -O3 -I .. -DNDEBUG benchmark.cpp -o benchmark && time ./benchmark >/dev/null
echo "Fixed size 3x3, column-major, with asserts"
$CXX -O3 -I .. benchmark.cpp -o benchmark && time ./benchmark >/dev/null
echo "Fixed size 3x3, row-major, -DNDEBUG"
$CXX -O3 -I .. -DEIGEN_DEFAULT_TO_ROW_MAJOR -DNDEBUG benchmark.cpp -o benchmark && time ./benchmark >/dev/null
echo "Fixed size 3x3, row-major, with asserts"
$CXX -O3 -I .. -DEIGEN_DEFAULT_TO_ROW_MAJOR benchmark.cpp -o benchmark && time ./benchmark >/dev/null
echo "Dynamic size 20x20, column-major, -DNDEBUG"
$CXX -O3 -I .. -DNDEBUG benchmarkX.cpp -o benchmarkX && time ./benchmarkX >/dev/null
echo "Dynamic size 20x20, column-major, with asserts"
$CXX -O3 -I .. benchmarkX.cpp -o benchmarkX && time ./benchmarkX >/dev/null
echo "Dynamic size 20x20, row-major, -DNDEBUG"
$CXX -O3 -I .. -DEIGEN_DEFAULT_TO_ROW_MAJOR -DNDEBUG benchmarkX.cpp -o benchmarkX && time ./benchmarkX >/dev/null
echo "Dynamic size 20x20, row-major, with asserts"
$CXX -O3 -I .. -DEIGEN_DEFAULT_TO_ROW_MAJOR benchmarkX.cpp -o benchmarkX && time ./benchmarkX >/dev/null
