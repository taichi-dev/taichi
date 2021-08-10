
This folder contains a couple of benchmark utities and Eigen benchmarks.

****************************
* bench_multi_compilers.sh *
****************************

This script allows to run a benchmark on a set of different compilers/compiler options.
It takes two arguments:
 - a file defining the list of the compilers with their options
 - the .cpp file of the benchmark

Examples:

$ ./bench_multi_compilers.sh basicbench.cxxlist basicbenchmark.cpp

    g++-4.1 -O3 -DNDEBUG -finline-limit=10000
    3d-3x3   /   4d-4x4   /   Xd-4x4   /   Xd-20x20   /
    0.271102   0.131416   0.422322   0.198633
    0.201658   0.102436   0.397566   0.207282

    g++-4.2 -O3 -DNDEBUG -finline-limit=10000
    3d-3x3   /   4d-4x4   /   Xd-4x4   /   Xd-20x20   /
    0.107805   0.0890579   0.30265   0.161843
    0.127157   0.0712581   0.278341   0.191029

    g++-4.3 -O3 -DNDEBUG -finline-limit=10000
    3d-3x3   /   4d-4x4   /   Xd-4x4   /   Xd-20x20   /
    0.134318   0.105291   0.3704   0.180966
    0.137703   0.0732472   0.31225   0.202204

    icpc -fast -DNDEBUG -fno-exceptions -no-inline-max-size
    3d-3x3   /   4d-4x4   /   Xd-4x4   /   Xd-20x20   /
    0.226145   0.0941319   0.371873   0.159433
    0.109302   0.0837538   0.328102   0.173891


$ ./bench_multi_compilers.sh ompbench.cxxlist ompbenchmark.cpp

    g++-4.2 -O3 -DNDEBUG -finline-limit=10000 -fopenmp
    double, fixed-size 4x4: 0.00165105s  0.0778739s
    double, 32x32: 0.0654769s 0.075289s  => x0.869674 (2)
    double, 128x128: 0.054148s 0.0419669s  => x1.29025 (2)
    double, 512x512: 0.913799s 0.428533s  => x2.13239 (2)
    double, 1024x1024: 14.5972s 9.3542s  => x1.5605 (2)

    icpc -fast -DNDEBUG -fno-exceptions -no-inline-max-size -openmp
    double, fixed-size 4x4: 0.000589848s  0.019949s
    double, 32x32: 0.0682781s 0.0449722s  => x1.51823 (2)
    double, 128x128: 0.0547509s 0.0435519s  => x1.25714 (2)
    double, 512x512: 0.829436s 0.424438s  => x1.9542 (2)
    double, 1024x1024: 14.5243s 10.7735s  => x1.34815 (2)



