ma: kernel_malloc.cu
	nvcc kernel_malloc.cu -std=c++14 -O3 -o ma

rand_nvcc: rand.cu
	nvcc rand.cu -o rand_nvcc

rand_clang: rand.cu
	clang++-8 -std=c++14 rand.cu -o rand_clang --cuda-gpu-arch=sm_61 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread

assert: assert.cu
	clang++-8 -std=c++14 assert.cu -S -emit-llvm --cuda-gpu-arch=sm_61

gpu_memory_bound: gpu_memory_bound.cu
	nvcc -std=c++14 gpu_memory_bound.cu -O3 -o gpu_memory_bound --gpu-architecture=compute_61 --gpu-code=sm_61,compute_61

gpu_reduction: gpu_reduction.cu
	nvcc -std=c++14 gpu_reduction.cu -O3 -o gpu_reduction --gpu-architecture=compute_61 --gpu-code=sm_61,compute_61

ldg:
	clang++-8 -std=c++14 ldg.cu -S -emit-llvm --cuda-gpu-arch=sm_61

cpu_memory_bound: cpu_memory_bound.cpp
	g++ -std=c++14 cpu_memory_bound.cpp -O3 -o cpu_memory_bound

cuda_atomic_reduce: cuda_atomic_reduce.cu
	nvcc -std=c++14 cuda_atomic_reduce.cu -O3 -o -cuda_atomic_reduce

laplace: laplace.cu
	nvcc -std=c++14 laplace.cu -O3 -o laplace
