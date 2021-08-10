// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Copyright (C) 2014 Navdeep Jaitly <ndjaitly@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_cuda
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;
typedef Tensor<float, 1>::DimensionPair DimPair;

template<int DataLayout>
void test_cuda_contraction(int m_size, int k_size, int n_size)
{
  std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size << ")" << std::endl;
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  Tensor<float, 2, DataLayout> t_left(m_size, k_size);
  Tensor<float, 2, DataLayout> t_right(k_size, n_size);
  Tensor<float, 2, DataLayout> t_result(m_size, n_size);
  Tensor<float, 2, DataLayout> t_result_gpu(m_size, n_size);
  Eigen::array<DimPair, 1> dims(DimPair(1, 0));

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size()  * sizeof(float);
  std::size_t t_right_bytes = t_right.size() * sizeof(float);
  std::size_t t_result_bytes = t_result.size() * sizeof(float);

  float* d_t_left;
  float* d_t_right;
  float* d_t_result;

  cudaMalloc((void**)(&d_t_left), t_left_bytes);
  cudaMalloc((void**)(&d_t_right), t_right_bytes);
  cudaMalloc((void**)(&d_t_result), t_result_bytes);

  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> >
      gpu_t_left(d_t_left, Eigen::array<int, 2>(m_size, k_size));
  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> >
      gpu_t_right(d_t_right, Eigen::array<int, 2>(k_size, n_size));
  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> >
      gpu_t_result(d_t_result, Eigen::array<int, 2>(m_size, n_size));


  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
  t_result = t_left.contract(t_right, dims);

  cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
  for (DenseIndex i = 0; i < t_result.size(); i++) {
    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
      continue;
    }
    std::cout << "mismatch detected at index " << i << ": " << t_result(i)
              << " vs " <<  t_result_gpu(i) << std::endl;
    assert(false);
  }

  cudaFree((void*)d_t_left);
  cudaFree((void*)d_t_right);
  cudaFree((void*)d_t_result);
}


template<int DataLayout>
void test_scalar(int m_size, int k_size, int n_size)
{
  std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size << ")" << std::endl;
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  Tensor<float, 2, DataLayout> t_left(m_size, k_size);
  Tensor<float, 2, DataLayout> t_right(k_size, n_size);
  Tensor<float, 0, DataLayout> t_result;
  Tensor<float, 0, DataLayout> t_result_gpu;
  Eigen::array<DimPair, 2> dims(DimPair(0, 0), DimPair(1, 1));

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size()  * sizeof(float);
  std::size_t t_right_bytes = t_right.size() * sizeof(float);
  std::size_t t_result_bytes = sizeof(float);

  float* d_t_left;
  float* d_t_right;
  float* d_t_result;

  cudaMalloc((void**)(&d_t_left), t_left_bytes);
  cudaMalloc((void**)(&d_t_right), t_right_bytes);
  cudaMalloc((void**)(&d_t_result), t_result_bytes);

  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> >
      gpu_t_left(d_t_left, m_size, k_size);
  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> >
      gpu_t_right(d_t_right, k_size, n_size);
  Eigen::TensorMap<Eigen::Tensor<float, 0, DataLayout> >
      gpu_t_result(d_t_result);

  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
  t_result = t_left.contract(t_right, dims);

  cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
  if (fabs(t_result() - t_result_gpu()) > 1e-4f &&
      !Eigen::internal::isApprox(t_result(), t_result_gpu(), 1e-4f)) {
    std::cout << "mismatch detected: " << t_result()
              << " vs " <<  t_result_gpu() << std::endl;
    assert(false);
  }

  cudaFree((void*)d_t_left);
  cudaFree((void*)d_t_right);
  cudaFree((void*)d_t_result);
}


template<int DataLayout>
void test_cuda_contraction_m() {
  for (int k = 32; k < 256; k++) {
    test_cuda_contraction<ColMajor>(k, 128, 128);
    test_cuda_contraction<RowMajor>(k, 128, 128);
  }
}

template<int DataLayout>
void test_cuda_contraction_k() {
  for (int k = 32; k < 256; k++) {
    test_cuda_contraction<ColMajor>(128, k, 128);
    test_cuda_contraction<RowMajor>(128, k, 128);
  }
}

template<int DataLayout>
void test_cuda_contraction_n() {
  for (int k = 32; k < 256; k++) {
    test_cuda_contraction<ColMajor>(128, 128, k);
    test_cuda_contraction<RowMajor>(128, 128, k);
  }
}


template<int DataLayout>
void test_cuda_contraction_sizes() {
  int m_sizes[] = { 31,  39,   63,   64,   65,
                   127, 129,  255,  257 , 511,
                   512, 513, 1023, 1024, 1025};

  int n_sizes[] = { 31,  39,   63,   64,   65,
                   127, 129,  255,  257,  511,
                   512, 513, 1023, 1024, 1025};

  int k_sizes[] = {  31,   39,  63,  64,   65,
                     95,   96, 127, 129,  255,
                    257,  511, 512, 513, 1023,
                   1024, 1025};

  for (int i = 0; i < 15; i++) {
    for (int j = 0; j < 15; j++) {
      for (int k = 0; k < 17; k++) {
        test_cuda_contraction<DataLayout>(m_sizes[i], n_sizes[j], k_sizes[k]);
      }
    }
  }
}

void test_cxx11_tensor_cuda()
{
  CALL_SUBTEST_1(test_cuda_contraction<ColMajor>(128, 128, 128));
  CALL_SUBTEST_1(test_cuda_contraction<RowMajor>(128, 128, 128));

  CALL_SUBTEST_1(test_scalar<ColMajor>(128, 128, 128));
  CALL_SUBTEST_1(test_scalar<RowMajor>(128, 128, 128));

  CALL_SUBTEST_2(test_cuda_contraction_m<ColMajor>());
  CALL_SUBTEST_3(test_cuda_contraction_m<RowMajor>());

  CALL_SUBTEST_4(test_cuda_contraction_k<ColMajor>());
  CALL_SUBTEST_5(test_cuda_contraction_k<RowMajor>());

  CALL_SUBTEST_6(test_cuda_contraction_n<ColMajor>());
  CALL_SUBTEST_7(test_cuda_contraction_n<RowMajor>());

  CALL_SUBTEST_8(test_cuda_contraction_sizes<ColMajor>());
  CALL_SUBTEST_9(test_cuda_contraction_sizes<RowMajor>());
}
