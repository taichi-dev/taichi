// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
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
#define EIGEN_TEST_FUNC cxx11_tensor_broadcast_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

static void test_broadcast_sycl(const Eigen::SyclDevice &sycl_device){

  // BROADCAST test:
  array<int, 4> in_range   = {{2, 3, 5, 7}};
  array<int, 4> broadcasts = {{2, 3, 1, 4}};
  array<int, 4> out_range;  // = in_range * broadcasts
  for (size_t i = 0; i < out_range.size(); ++i)
    out_range[i] = in_range[i] * broadcasts[i];

  Tensor<float, 4>  input(in_range);
  Tensor<float, 4> out(out_range);

  for (size_t i = 0; i < in_range.size(); ++i)
    VERIFY_IS_EQUAL(out.dimension(i), out_range[i]);


  for (int i = 0; i < input.size(); ++i)
    input(i) = static_cast<float>(i);

  float * gpu_in_data  = static_cast<float*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(float)));
  float * gpu_out_data  = static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));

  TensorMap<Tensor<float, 4>>  gpu_in(gpu_in_data, in_range);
  TensorMap<Tensor<float, 4>> gpu_out(gpu_out_data, out_range);
  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(float));
  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 28; ++l) {
          VERIFY_IS_APPROX(input(i%2,j%3,k%5,l%7), out(i,j,k,l));
        }
      }
    }
  }
  printf("Broadcast Test Passed\n");
  sycl_device.deallocate(gpu_in_data);
  sycl_device.deallocate(gpu_out_data);
}

void test_cxx11_tensor_broadcast_sycl() {
  cl::sycl::gpu_selector s;
  Eigen::SyclDevice sycl_device(s);
  CALL_SUBTEST(test_broadcast_sycl(sycl_device));
}
