
#ifndef EIGEN_TEST_CUDA_COMMON_H
#define EIGEN_TEST_CUDA_COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

#ifndef __CUDACC__
dim3 threadIdx, blockDim, blockIdx;
#endif

template<typename Kernel, typename Input, typename Output>
void run_on_cpu(const Kernel& ker, int n, const Input& in, Output& out)
{
  for(int i=0; i<n; i++)
    ker(i, in.data(), out.data());
}


template<typename Kernel, typename Input, typename Output>
__global__
void run_on_cuda_meta_kernel(const Kernel ker, int n, const Input* in, Output* out)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i<n) {
    ker(i, in, out);
  }
}


template<typename Kernel, typename Input, typename Output>
void run_on_cuda(const Kernel& ker, int n, const Input& in, Output& out)
{
  typename Input::Scalar*  d_in;
  typename Output::Scalar* d_out;
  std::ptrdiff_t in_bytes  = in.size()  * sizeof(typename Input::Scalar);
  std::ptrdiff_t out_bytes = out.size() * sizeof(typename Output::Scalar);
  
  cudaMalloc((void**)(&d_in),  in_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);
  
  cudaMemcpy(d_in,  in.data(),  in_bytes,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out.data(), out_bytes, cudaMemcpyHostToDevice);
  
  // Simple and non-optimal 1D mapping assuming n is not too large
  // That's only for unit testing!
  dim3 Blocks(128);
  dim3 Grids( (n+int(Blocks.x)-1)/int(Blocks.x) );

  cudaThreadSynchronize();
  run_on_cuda_meta_kernel<<<Grids,Blocks>>>(ker, n, d_in, d_out);
  cudaThreadSynchronize();
  
  // check inputs have not been modified
  cudaMemcpy(const_cast<typename Input::Scalar*>(in.data()),  d_in,  in_bytes,  cudaMemcpyDeviceToHost);
  cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
  
  cudaFree(d_in);
  cudaFree(d_out);
}


template<typename Kernel, typename Input, typename Output>
void run_and_compare_to_cuda(const Kernel& ker, int n, const Input& in, Output& out)
{
  Input  in_ref,  in_cuda;
  Output out_ref, out_cuda;
  #ifndef __CUDA_ARCH__
  in_ref = in_cuda = in;
  out_ref = out_cuda = out;
  #endif
  run_on_cpu (ker, n, in_ref,  out_ref);
  run_on_cuda(ker, n, in_cuda, out_cuda);
  #ifndef __CUDA_ARCH__
  VERIFY_IS_APPROX(in_ref, in_cuda);
  VERIFY_IS_APPROX(out_ref, out_cuda);
  #endif
}


void ei_test_init_cuda()
{
  int device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  std::cout << "CUDA device info:\n";
  std::cout << "  name:                        " << deviceProp.name << "\n";
  std::cout << "  capability:                  " << deviceProp.major << "." << deviceProp.minor << "\n";
  std::cout << "  multiProcessorCount:         " << deviceProp.multiProcessorCount << "\n";
  std::cout << "  maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << "\n";
  std::cout << "  warpSize:                    " << deviceProp.warpSize << "\n";
  std::cout << "  regsPerBlock:                " << deviceProp.regsPerBlock << "\n";
  std::cout << "  concurrentKernels:           " << deviceProp.concurrentKernels << "\n";
  std::cout << "  clockRate:                   " << deviceProp.clockRate << "\n";
  std::cout << "  canMapHostMemory:            " << deviceProp.canMapHostMemory << "\n";
  std::cout << "  computeMode:                 " << deviceProp.computeMode << "\n";
}

#endif // EIGEN_TEST_CUDA_COMMON_H
