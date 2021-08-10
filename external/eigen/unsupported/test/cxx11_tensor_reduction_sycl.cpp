// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_reduction_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>



static void test_full_reductions_sycl(const Eigen::SyclDevice&  sycl_device) {

  const int num_rows = 452;
  const int num_cols = 765;
  array<int, 2> tensorRange = {{num_rows, num_cols}};

  Tensor<float, 2> in(tensorRange);
  Tensor<float, 0> full_redux;
  Tensor<float, 0> full_redux_gpu;

  in.setRandom();

  full_redux = in.sum();

  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
  float* gpu_out_data =(float*)sycl_device.allocate(sizeof(float));

  TensorMap<Tensor<float, 2> >  in_gpu(gpu_in_data, tensorRange);
  TensorMap<Tensor<float, 0> >  out_gpu(gpu_out_data);

  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
  out_gpu.device(sycl_device) = in_gpu.sum();
  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data, sizeof(float));
  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

static void test_first_dim_reductions_sycl(const Eigen::SyclDevice& sycl_device) {

  int dim_x = 145;
  int dim_y = 1;
  int dim_z = 67;

  array<int, 3> tensorRange = {{dim_x, dim_y, dim_z}};
  Eigen::array<int, 1> red_axis;
  red_axis[0] = 0;
  array<int, 2> reduced_tensorRange = {{dim_y, dim_z}};

  Tensor<float, 3> in(tensorRange);
  Tensor<float, 2> redux(reduced_tensorRange);
  Tensor<float, 2> redux_gpu(reduced_tensorRange);

  in.setRandom();

  redux= in.sum(red_axis);

  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
  float* gpu_out_data = static_cast<float*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(float)));

  TensorMap<Tensor<float, 3> >  in_gpu(gpu_in_data, tensorRange);
  TensorMap<Tensor<float, 2> >  out_gpu(gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(float));

  // Check that the CPU and GPU reductions return the same result.
  for(int j=0; j<reduced_tensorRange[0]; j++ )
    for(int k=0; k<reduced_tensorRange[1]; k++ )
      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

static void test_last_dim_reductions_sycl(const Eigen::SyclDevice &sycl_device) {

  int dim_x = 567;
  int dim_y = 1;
  int dim_z = 47;

  array<int, 3> tensorRange = {{dim_x, dim_y, dim_z}};
  Eigen::array<int, 1> red_axis;
  red_axis[0] = 2;
  array<int, 2> reduced_tensorRange = {{dim_x, dim_y}};

  Tensor<float, 3> in(tensorRange);
  Tensor<float, 2> redux(reduced_tensorRange);
  Tensor<float, 2> redux_gpu(reduced_tensorRange);

  in.setRandom();

  redux= in.sum(red_axis);

  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
  float* gpu_out_data = static_cast<float*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(float)));

  TensorMap<Tensor<float, 3> >  in_gpu(gpu_in_data, tensorRange);
  TensorMap<Tensor<float, 2> >  out_gpu(gpu_out_data, reduced_tensorRange);

  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(float));
  // Check that the CPU and GPU reductions return the same result.
  for(int j=0; j<reduced_tensorRange[0]; j++ )
    for(int k=0; k<reduced_tensorRange[1]; k++ )
      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));

  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);

}

void test_cxx11_tensor_reduction_sycl() {
  cl::sycl::gpu_selector s;
  Eigen::SyclDevice sycl_device(s);
  CALL_SUBTEST((test_full_reductions_sycl(sycl_device)));
  CALL_SUBTEST((test_first_dim_reductions_sycl(sycl_device)));
  CALL_SUBTEST((test_last_dim_reductions_sycl(sycl_device)));

}
