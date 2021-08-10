// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_FUNC cxx11_tensor_complex_cwise_ops
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

template<typename T>
void test_cuda_complex_cwise_ops() {
  const int kNumItems = 2;
  std::size_t complex_bytes = kNumItems * sizeof(std::complex<T>);

  std::complex<T>* d_in1;
  std::complex<T>* d_in2;
  std::complex<T>* d_out;
  cudaMalloc((void**)(&d_in1), complex_bytes);
  cudaMalloc((void**)(&d_in2), complex_bytes);
  cudaMalloc((void**)(&d_out), complex_bytes);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in1(
      d_in1, kNumItems);
  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in2(
      d_in2, kNumItems);
  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_out(
      d_out, kNumItems);

  const std::complex<T> a(3.14f, 2.7f);
  const std::complex<T> b(-10.6f, 1.4f);

  gpu_in1.device(gpu_device) = gpu_in1.constant(a);
  gpu_in2.device(gpu_device) = gpu_in2.constant(b);

  enum CwiseOp {
    Add = 0,
    Sub,
    Mul,
    Div
  };

  Tensor<std::complex<T>, 1, 0, int> actual(kNumItems);
  for (int op = Add; op <= Div; op++) {
    std::complex<T> expected;
    switch (static_cast<CwiseOp>(op)) {
      case Add:
        gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
        expected = a + b;
        break;
      case Sub:
        gpu_out.device(gpu_device) = gpu_in1 - gpu_in2;
        expected = a - b;
        break;
      case Mul:
        gpu_out.device(gpu_device) = gpu_in1 * gpu_in2;
        expected = a * b;
        break;
      case Div:
        gpu_out.device(gpu_device) = gpu_in1 / gpu_in2;
        expected = a / b;
        break;
    }
    assert(cudaMemcpyAsync(actual.data(), d_out, complex_bytes, cudaMemcpyDeviceToHost,
                           gpu_device.stream()) == cudaSuccess);
    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

    for (int i = 0; i < kNumItems; ++i) {
      VERIFY_IS_APPROX(actual(i), expected);
    }
  }

  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}


void test_cxx11_tensor_complex_cwise_ops()
{
  CALL_SUBTEST(test_cuda_complex_cwise_ops<float>());
  CALL_SUBTEST(test_cuda_complex_cwise_ops<double>());
}
