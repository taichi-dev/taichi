sample: sample.cpp kernel.ll experiment.cu
	llc -mcpu=sm_60 kernel.ll -o kernel.ptx
	clang++-7 sample.cpp -o sample -O2 -g -I/usr/local/cuda/include -lcuda
	clang++-7 experiment.cu --cuda-gpu-arch=sm_35 -std=c++14 -S -emit-llvm
