// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_FUNC cxx11_tensor_complex
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

void test_cuda_nullary() {
  Tensor<std::complex<float>, 1, 0, int> in1(2);
  Tensor<std::complex<float>, 1, 0, int> in2(2);
  in1.setRandom();
  in2.setRandom();

  std::size_t float_bytes = in1.size() * sizeof(float);
  std::size_t complex_bytes = in1.size() * sizeof(std::complex<float>);

  std::complex<float>* d_in1;
  std::complex<float>* d_in2;
  float* d_out2;
  cudaMalloc((void**)(&d_in1), complex_bytes);
  cudaMalloc((void**)(&d_in2), complex_bytes);
  cudaMalloc((void**)(&d_out2), float_bytes);
  cudaMemcpy(d_in1, in1.data(), complex_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2.data(), complex_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in1(
      d_in1, 2);
  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in2(
      d_in2, 2);
  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_out2(
      d_out2, 2);

  gpu_in1.device(gpu_device) = gpu_in1.constant(std::complex<float>(3.14f, 2.7f));
  gpu_out2.device(gpu_device) = gpu_in2.abs();

  Tensor<std::complex<float>, 1, 0, int> new1(2);
  Tensor<float, 1, 0, int> new2(2);

  assert(cudaMemcpyAsync(new1.data(), d_in1, complex_bytes, cudaMemcpyDeviceToHost,
                         gpu_device.stream()) == cudaSuccess);
  assert(cudaMemcpyAsync(new2.data(), d_out2, float_bytes, cudaMemcpyDeviceToHost,
                         gpu_device.stream()) == cudaSuccess);

  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 2; ++i) {
    VERIFY_IS_APPROX(new1(i), std::complex<float>(3.14f, 2.7f));
    VERIFY_IS_APPROX(new2(i), std::abs(in2(i)));
  }

  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out2);
}


static void test_cuda_sum_reductions() {

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  const int num_rows = internal::random<int>(1024, 5*1024);
  const int num_cols = internal::random<int>(1024, 5*1024);

  Tensor<std::complex<float>, 2> in(num_rows, num_cols);
  in.setRandom();

  Tensor<std::complex<float>, 0> full_redux;
  full_redux = in.sum();

  std::size_t in_bytes = in.size() * sizeof(std::complex<float>);
  std::size_t out_bytes = full_redux.size() * sizeof(std::complex<float>);
  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);

  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);

  out_gpu.device(gpu_device) = in_gpu.sum();

  Tensor<std::complex<float>, 0> full_redux_gpu;
  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
  gpu_device.synchronize();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());

  gpu_device.deallocate(gpu_in_ptr);
  gpu_device.deallocate(gpu_out_ptr);
}


static void test_cuda_product_reductions() {

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  const int num_rows = internal::random<int>(1024, 5*1024);
  const int num_cols = internal::random<int>(1024, 5*1024);

  Tensor<std::complex<float>, 2> in(num_rows, num_cols);
  in.setRandom();

  Tensor<std::complex<float>, 0> full_redux;
  full_redux = in.prod();

  std::size_t in_bytes = in.size() * sizeof(std::complex<float>);
  std::size_t out_bytes = full_redux.size() * sizeof(std::complex<float>);
  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);

  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);

  out_gpu.device(gpu_device) = in_gpu.prod();

  Tensor<std::complex<float>, 0> full_redux_gpu;
  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
  gpu_device.synchronize();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());

  gpu_device.deallocate(gpu_in_ptr);
  gpu_device.deallocate(gpu_out_ptr);
}


void test_cxx11_tensor_complex()
{
  CALL_SUBTEST(test_cuda_nullary());
  CALL_SUBTEST(test_cuda_sum_reductions());
  CALL_SUBTEST(test_cuda_product_reductions());
}
