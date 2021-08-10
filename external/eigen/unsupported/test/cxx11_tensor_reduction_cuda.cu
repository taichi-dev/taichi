// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_reduction_cuda
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>


template<typename Type, int DataLayout>
static void test_full_reductions() {

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  const int num_rows = internal::random<int>(1024, 5*1024);
  const int num_cols = internal::random<int>(1024, 5*1024);

  Tensor<Type, 2, DataLayout> in(num_rows, num_cols);
  in.setRandom();

  Tensor<Type, 0, DataLayout> full_redux;
  full_redux = in.sum();

  std::size_t in_bytes = in.size() * sizeof(Type);
  std::size_t out_bytes = full_redux.size() * sizeof(Type);
  Type* gpu_in_ptr = static_cast<Type*>(gpu_device.allocate(in_bytes));
  Type* gpu_out_ptr = static_cast<Type*>(gpu_device.allocate(out_bytes));
  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);

  TensorMap<Tensor<Type, 2, DataLayout> > in_gpu(gpu_in_ptr, num_rows, num_cols);
  TensorMap<Tensor<Type, 0, DataLayout> > out_gpu(gpu_out_ptr);

  out_gpu.device(gpu_device) = in_gpu.sum();

  Tensor<Type, 0, DataLayout> full_redux_gpu;
  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
  gpu_device.synchronize();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());

  gpu_device.deallocate(gpu_in_ptr);
  gpu_device.deallocate(gpu_out_ptr);
}

template<typename Type, int DataLayout>
static void test_first_dim_reductions() {
  int dim_x = 33;
  int dim_y = 1;
  int dim_z = 128;

  Tensor<Type, 3, DataLayout> in(dim_x, dim_y, dim_z);
  in.setRandom();

  Eigen::array<int, 1> red_axis;
  red_axis[0] = 0;
  Tensor<Type, 2, DataLayout> redux = in.sum(red_axis);

  // Create device
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice dev(&stream);
  
  // Create data(T)
  Type* in_data = (Type*)dev.allocate(dim_x*dim_y*dim_z*sizeof(Type));
  Type* out_data = (Type*)dev.allocate(dim_z*dim_y*sizeof(Type));
  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_y, dim_z);
  
  // Perform operation
  dev.memcpyHostToDevice(in_data, in.data(), in.size()*sizeof(Type));
  gpu_out.device(dev) = gpu_in.sum(red_axis);
  gpu_out.device(dev) += gpu_in.sum(red_axis);
  Tensor<Type, 2, DataLayout> redux_gpu(dim_y, dim_z);
  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
  dev.synchronize();

  // Check that the CPU and GPU reductions return the same result.
  for (int i = 0; i < gpu_out.size(); ++i) {
    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
  }

  dev.deallocate(in_data);
  dev.deallocate(out_data);
}

template<typename Type, int DataLayout>
static void test_last_dim_reductions() {
  int dim_x = 128;
  int dim_y = 1;
  int dim_z = 33;

  Tensor<Type, 3, DataLayout> in(dim_x, dim_y, dim_z);
  in.setRandom();

  Eigen::array<int, 1> red_axis;
  red_axis[0] = 2;
  Tensor<Type, 2, DataLayout> redux = in.sum(red_axis);

  // Create device
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice dev(&stream);
  
  // Create data
  Type* in_data = (Type*)dev.allocate(dim_x*dim_y*dim_z*sizeof(Type));
  Type* out_data = (Type*)dev.allocate(dim_x*dim_y*sizeof(Type));
  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_x, dim_y);
  
  // Perform operation
  dev.memcpyHostToDevice(in_data, in.data(), in.size()*sizeof(Type));
  gpu_out.device(dev) = gpu_in.sum(red_axis);
  gpu_out.device(dev) += gpu_in.sum(red_axis);
  Tensor<Type, 2, DataLayout> redux_gpu(dim_x, dim_y);
  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
  dev.synchronize();

  // Check that the CPU and GPU reductions return the same result.
  for (int i = 0; i < gpu_out.size(); ++i) {
    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
  }

  dev.deallocate(in_data);
  dev.deallocate(out_data);
}


void test_cxx11_tensor_reduction_cuda() {
  CALL_SUBTEST_1((test_full_reductions<float, ColMajor>()));
  CALL_SUBTEST_1((test_full_reductions<double, ColMajor>()));
  CALL_SUBTEST_2((test_full_reductions<float, RowMajor>()));
  CALL_SUBTEST_2((test_full_reductions<double, RowMajor>()));
  
  CALL_SUBTEST_3((test_first_dim_reductions<float, ColMajor>()));
  CALL_SUBTEST_3((test_first_dim_reductions<double, ColMajor>()));
  CALL_SUBTEST_4((test_first_dim_reductions<float, RowMajor>()));
// Outer reductions of doubles aren't supported just yet.  					      
//  CALL_SUBTEST_4((test_first_dim_reductions<double, RowMajor>()))

  CALL_SUBTEST_5((test_last_dim_reductions<float, ColMajor>()));
// Outer reductions of doubles aren't supported just yet.  					      
//  CALL_SUBTEST_5((test_last_dim_reductions<double, ColMajor>()));
  CALL_SUBTEST_6((test_last_dim_reductions<float, RowMajor>()));
  CALL_SUBTEST_6((test_last_dim_reductions<double, RowMajor>()));
}
