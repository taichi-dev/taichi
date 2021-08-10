// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_random_cuda
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include "main.h"
#include <Eigen/CXX11/Tensor>


void test_cuda_random_uniform()
{
  Tensor<float, 2> out(72,97);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_out;
  cudaMalloc((void**)(&d_out), out_bytes);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);

  gpu_out.device(gpu_device) = gpu_out.random();

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  // For now we just check thes code doesn't crash.
  // TODO: come up with a valid test of randomness
}


void test_cuda_random_normal()
{
  Tensor<float, 2> out(72,97);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_out;
  cudaMalloc((void**)(&d_out), out_bytes);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);

  Eigen::internal::NormalRandomGenerator<float> gen(true);
  gpu_out.device(gpu_device) = gpu_out.random(gen);

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
}

static void test_complex()
{
  Tensor<std::complex<float>, 1> vec(6);
  vec.setRandom();

  // Fixme: we should check that the generated numbers follow a uniform
  // distribution instead.
  for (int i = 1; i < 6; ++i) {
    VERIFY_IS_NOT_EQUAL(vec(i), vec(i-1));
  }
}


void test_cxx11_tensor_random_cuda()
{
  CALL_SUBTEST(test_cuda_random_uniform());
  CALL_SUBTEST(test_cuda_random_normal());
  CALL_SUBTEST(test_complex());
}
