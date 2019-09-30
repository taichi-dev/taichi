sample: sample.cpp kernel.ll
	llc -mcpu=sm_60 kernel.ll -o kernel.ptx
	clang++-7 sample.cpp -o sample -O2 -g -I/usr/local/cuda/include -lcuda
