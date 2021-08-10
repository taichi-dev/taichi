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
#define EIGEN_TEST_FUNC cxx11_tensor_forced_eval_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

void test_forced_eval_sycl(const Eigen::SyclDevice &sycl_device) {

  int sizeDim1 = 100;
  int sizeDim2 = 200;
  int sizeDim3 = 200;
  Eigen::array<int, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  Eigen::Tensor<float, 3> in1(tensorRange);
  Eigen::Tensor<float, 3> in2(tensorRange);
  Eigen::Tensor<float, 3> out(tensorRange);

  float * gpu_in1_data  = static_cast<float*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(float)));
  float * gpu_in2_data  = static_cast<float*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(float)));
  float * gpu_out_data =  static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));

  in1 = in1.random() + in1.constant(10.0f);
  in2 = in2.random() + in2.constant(10.0f);

  // creating TensorMap from tensor
  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_in1(gpu_in1_data, tensorRange);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_in2(gpu_in2_data, tensorRange);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_out(gpu_out_data, tensorRange);
  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(float));
  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in1.dimensions().TotalSize())*sizeof(float));
  /// c=(a+b)*b
  gpu_out.device(sycl_device) =(gpu_in1 + gpu_in2).eval() * gpu_in2;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i, j, k),
                         (in1(i, j, k) + in2(i, j, k)) * in2(i, j, k));
      }
    }
  }
  printf("(a+b)*b Test Passed\n");
  sycl_device.deallocate(gpu_in1_data);
  sycl_device.deallocate(gpu_in2_data);
  sycl_device.deallocate(gpu_out_data);

}

void test_cxx11_tensor_forced_eval_sycl() {
  cl::sycl::gpu_selector s;
  Eigen::SyclDevice sycl_device(s);
  CALL_SUBTEST(test_forced_eval_sycl(sycl_device));
}
