// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_of_float16_cuda
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

template<typename>
void test_cuda_numext() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;

  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
  bool* d_res_half = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
  bool* d_res_float = (bool*)gpu_device.allocate(num_elem * sizeof(bool));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
      d_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_half(
      d_res_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_float(
      d_res_float, num_elem);

  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
  gpu_res_float.device(gpu_device) = gpu_float.unaryExpr(Eigen::internal::scalar_isnan_op<float>());
  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().unaryExpr(Eigen::internal::scalar_isnan_op<Eigen::half>());

  Tensor<bool, 1> half_prec(num_elem);
  Tensor<bool, 1> full_prec(num_elem);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(bool));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(bool));
  gpu_device.synchronize();

  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking numext " << i << std::endl;
    VERIFY_IS_EQUAL(full_prec(i), half_prec(i));
  }

  gpu_device.deallocate(d_float);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}


#ifdef EIGEN_HAS_CUDA_FP16

template<typename>
void test_cuda_conversion() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;
  
  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
      d_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
      d_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
      d_conv, num_elem);

  gpu_float.device(gpu_device) = gpu_float.random();
  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
  gpu_conv.device(gpu_device) = gpu_half.cast<float>();

  Tensor<float, 1> initial(num_elem);
  Tensor<float, 1> final(num_elem);
  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));

  for (int i = 0; i < num_elem; ++i) {
    VERIFY_IS_APPROX(initial(i), final(i));
  }

  gpu_device.deallocate(d_float);
  gpu_device.deallocate(d_half);
  gpu_device.deallocate(d_conv);
}

template<typename>
void test_cuda_unary() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;

  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
      d_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
      d_res_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
      d_res_float, num_elem);

  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
  gpu_res_float.device(gpu_device) = gpu_float.abs();
  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().cast<float>();

  Tensor<float, 1> half_prec(num_elem);
  Tensor<float, 1> full_prec(num_elem);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
  gpu_device.synchronize();

  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking unary " << i << std::endl;
    VERIFY_IS_APPROX(full_prec(i), half_prec(i));
  }

  gpu_device.deallocate(d_float);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}

template<typename>
void test_cuda_elementwise() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;

  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(
      d_float1, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(
      d_float2, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
      d_res_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
      d_res_float, num_elem);

  gpu_float1.device(gpu_device) = gpu_float1.random();
  gpu_float2.device(gpu_device) = gpu_float2.random();
  gpu_res_float.device(gpu_device) = (gpu_float1 + gpu_float2) * gpu_float1;
  gpu_res_half.device(gpu_device) = ((gpu_float1.cast<Eigen::half>() + gpu_float2.cast<Eigen::half>()) * gpu_float1.cast<Eigen::half>()).cast<float>();

  Tensor<float, 1> half_prec(num_elem);
  Tensor<float, 1> full_prec(num_elem);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
  gpu_device.synchronize();

  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking elemwise " << i << ": full prec = " << full_prec(i) << " vs half prec = " << half_prec(i) << std::endl;
    VERIFY_IS_APPROX(static_cast<Eigen::half>(full_prec(i)), static_cast<Eigen::half>(half_prec(i)));
  }

  gpu_device.deallocate(d_float1);
  gpu_device.deallocate(d_float2);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}

template<typename>
void test_cuda_trancendental() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;

  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float3 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  Eigen::half* d_res1_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
  Eigen::half* d_res1_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
  Eigen::half* d_res2_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
  Eigen::half* d_res2_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
  Eigen::half* d_res3_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
  Eigen::half* d_res3_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(d_float1, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(d_float2, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float3(d_float3, num_elem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_half(d_res1_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_float(d_res1_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_half(d_res2_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_float(d_res2_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_half(d_res3_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_float(d_res3_float, num_elem);

  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
  gpu_float2.device(gpu_device) = gpu_float2.random() + gpu_float1.constant(0.5f);
  gpu_float3.device(gpu_device) = gpu_float3.random();
  gpu_res1_float.device(gpu_device) = gpu_float1.exp().cast<Eigen::half>();
  gpu_res2_float.device(gpu_device) = gpu_float2.log().cast<Eigen::half>();
  gpu_res3_float.device(gpu_device) = gpu_float3.log1p().cast<Eigen::half>();

  gpu_res1_half.device(gpu_device) = gpu_float1.cast<Eigen::half>();
  gpu_res1_half.device(gpu_device) = gpu_res1_half.exp();

  gpu_res2_half.device(gpu_device) = gpu_float2.cast<Eigen::half>();
  gpu_res2_half.device(gpu_device) = gpu_res2_half.log();

  gpu_res3_half.device(gpu_device) = gpu_float3.cast<Eigen::half>();
  gpu_res3_half.device(gpu_device) = gpu_res3_half.log1p();

  Tensor<float, 1> input1(num_elem);
  Tensor<Eigen::half, 1> half_prec1(num_elem);
  Tensor<Eigen::half, 1> full_prec1(num_elem);
  Tensor<float, 1> input2(num_elem);
  Tensor<Eigen::half, 1> half_prec2(num_elem);
  Tensor<Eigen::half, 1> full_prec2(num_elem);
  Tensor<float, 1> input3(num_elem);
  Tensor<Eigen::half, 1> half_prec3(num_elem);
  Tensor<Eigen::half, 1> full_prec3(num_elem);
  gpu_device.memcpyDeviceToHost(input1.data(), d_float1, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(input2.data(), d_float2, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(input3.data(), d_float3, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res1_half, num_elem*sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(full_prec1.data(), d_res1_float, num_elem*sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res2_half, num_elem*sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(full_prec2.data(), d_res2_float, num_elem*sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(half_prec3.data(), d_res3_half, num_elem*sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(full_prec3.data(), d_res3_float, num_elem*sizeof(Eigen::half));
  gpu_device.synchronize();

  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking elemwise exp " << i << " input = " << input1(i) << " full = " << full_prec1(i) << " half = " << half_prec1(i) << std::endl;
    VERIFY_IS_APPROX(full_prec1(i), half_prec1(i));
  }
  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking elemwise log " << i << " input = " << input2(i) << " full = " << full_prec2(i) << " half = " << half_prec2(i) << std::endl;
    if(std::abs(input2(i)-1.f)<0.05f) // log lacks accurary nearby 1
      VERIFY_IS_APPROX(full_prec2(i)+Eigen::half(0.1f), half_prec2(i)+Eigen::half(0.1f));
    else
      VERIFY_IS_APPROX(full_prec2(i), half_prec2(i));
  }
  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking elemwise plog1 " << i << " input = " << input3(i) << " full = " << full_prec3(i) << " half = " << half_prec3(i) << std::endl;
    VERIFY_IS_APPROX(full_prec3(i), half_prec3(i));
  }
  gpu_device.deallocate(d_float1);
  gpu_device.deallocate(d_float2);
  gpu_device.deallocate(d_float3);
  gpu_device.deallocate(d_res1_half);
  gpu_device.deallocate(d_res1_float);
  gpu_device.deallocate(d_res2_half);
  gpu_device.deallocate(d_res2_float);
  gpu_device.deallocate(d_res3_float);
  gpu_device.deallocate(d_res3_half);
}

template<typename>
void test_cuda_contractions() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int rows = 23;
  int cols = 23;
  int num_elem = rows*cols;

  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));

  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
      d_float1, rows, cols);
  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
      d_float2, rows, cols);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_half(
      d_res_half, rows, cols);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_float(
      d_res_float, rows, cols);

  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
  gpu_float2.device(gpu_device) = gpu_float2.random() - gpu_float2.constant(0.5f);

  typedef Tensor<float, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims(DimPair(1, 0));
  gpu_res_float.device(gpu_device) = gpu_float1.contract(gpu_float2, dims).cast<Eigen::half>();
  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().contract(gpu_float2.cast<Eigen::half>(), dims);

  Tensor<Eigen::half, 2> half_prec(rows, cols);
  Tensor<Eigen::half, 2> full_prec(rows, cols);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(Eigen::half));
  gpu_device.synchronize();

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << "Checking contract " << i << " " << j << full_prec(i, j) << " " << half_prec(i, j) << std::endl;
      if (numext::abs(full_prec(i, j) - half_prec(i, j)) > Eigen::half(1e-2f)) {
        VERIFY_IS_APPROX(full_prec(i, j), half_prec(i, j));
      }
    }
  }

  gpu_device.deallocate(d_float1);
  gpu_device.deallocate(d_float2);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}

template<typename>
void test_cuda_reductions(int size1, int size2, int redux) {

   std::cout << "Reducing " << size1 << " by " << size2
             << " tensor along dim " << redux << std::endl; 

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = size1*size2;
  int result_size = (redux == 1 ? size1 : size2);

  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));

  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
      d_float1, size1, size2);
  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
      d_float2, size1, size2);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_half(
      d_res_half, result_size);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_float(
      d_res_float, result_size);

  gpu_float1.device(gpu_device) = gpu_float1.random() * 2.0f;
  gpu_float2.device(gpu_device) = gpu_float2.random() * 2.0f;

  Eigen::array<int, 1> redux_dim = {{redux}};
  gpu_res_float.device(gpu_device) = gpu_float1.sum(redux_dim).cast<Eigen::half>();
  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum(redux_dim);

  Tensor<Eigen::half, 1> half_prec(result_size);
  Tensor<Eigen::half, 1> full_prec(result_size);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, result_size*sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, result_size*sizeof(Eigen::half));
  gpu_device.synchronize();

  for (int i = 0; i < result_size; ++i) {
    std::cout << "EXPECTED " << full_prec(i) << " GOT " << half_prec(i) << std::endl;
    VERIFY_IS_APPROX(full_prec(i), half_prec(i));
  }

  gpu_device.deallocate(d_float1);
  gpu_device.deallocate(d_float2);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}

template<typename>
void test_cuda_reductions() {
  test_cuda_reductions<void>(13, 13, 0);
  test_cuda_reductions<void>(13, 13, 1);

  test_cuda_reductions<void>(35, 36, 0);
  test_cuda_reductions<void>(35, 36, 1);

  test_cuda_reductions<void>(36, 35, 0);
  test_cuda_reductions<void>(36, 35, 1);
}

template<typename>
void test_cuda_full_reductions() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int size = 13;
  int num_elem = size*size;

  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));

  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
      d_float1, size, size);
  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
      d_float2, size, size);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_half(
      d_res_half);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_float(
      d_res_float);

  gpu_float1.device(gpu_device) = gpu_float1.random();
  gpu_float2.device(gpu_device) = gpu_float2.random();

  gpu_res_float.device(gpu_device) = gpu_float1.sum().cast<Eigen::half>();
  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum();

  Tensor<Eigen::half, 0> half_prec;
  Tensor<Eigen::half, 0> full_prec;
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
  gpu_device.synchronize();

  VERIFY_IS_APPROX(full_prec(), half_prec());

  gpu_res_float.device(gpu_device) = gpu_float1.maximum().cast<Eigen::half>();
  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().maximum();
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
  gpu_device.synchronize();

  VERIFY_IS_APPROX(full_prec(), half_prec());

  gpu_device.deallocate(d_float1);
  gpu_device.deallocate(d_float2);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}

template<typename>
void test_cuda_forced_evals() {

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;

  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_half1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_half2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
      d_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half1(
      d_res_half1, num_elem);
 Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Unaligned> gpu_res_half2(
      d_res_half2, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
      d_res_float, num_elem);

  Eigen::array<int, 1> no_bcast;
  no_bcast[0] = 1;

  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
  gpu_res_float.device(gpu_device) = gpu_float.abs();
  gpu_res_half1.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().eval().cast<float>();
  gpu_res_half2.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().broadcast(no_bcast).eval().cast<float>();

  Tensor<float, 1> half_prec1(num_elem);
  Tensor<float, 1> half_prec2(num_elem);
  Tensor<float, 1> full_prec(num_elem);
  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res_half1, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res_half1, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
  gpu_device.synchronize();

  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking forced eval " << i << full_prec(i) << " vs " << half_prec1(i) << " vs " << half_prec2(i) << std::endl;
    VERIFY_IS_APPROX(full_prec(i), half_prec1(i));
    VERIFY_IS_APPROX(full_prec(i), half_prec2(i));
  }

  gpu_device.deallocate(d_float);
  gpu_device.deallocate(d_res_half1);
  gpu_device.deallocate(d_res_half2);
  gpu_device.deallocate(d_res_float);
}
#endif


void test_cxx11_tensor_of_float16_cuda()
{
  CALL_SUBTEST_1(test_cuda_numext<void>());

#ifdef EIGEN_HAS_CUDA_FP16
  CALL_SUBTEST_1(test_cuda_conversion<void>());
  CALL_SUBTEST_1(test_cuda_unary<void>());
  CALL_SUBTEST_1(test_cuda_elementwise<void>());
  CALL_SUBTEST_1(test_cuda_trancendental<void>());
  CALL_SUBTEST_2(test_cuda_contractions<void>());
  CALL_SUBTEST_3(test_cuda_reductions<void>());
  CALL_SUBTEST_4(test_cuda_full_reductions<void>());
  CALL_SUBTEST_5(test_cuda_forced_evals<void>());
#else
  std::cout << "Half floats are not supported by this version of cuda: skipping the test" << std::endl;
#endif
}
