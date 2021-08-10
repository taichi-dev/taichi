// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

void test_sycl_cpu(const Eigen::SyclDevice &sycl_device) {

  int sizeDim1 = 100;
  int sizeDim2 = 100;
  int sizeDim3 = 100;
  array<int, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  Tensor<float, 3> in1(tensorRange);
  Tensor<float, 3> in2(tensorRange);
  Tensor<float, 3> in3(tensorRange);
  Tensor<float, 3> out(tensorRange);

  in2 = in2.random();
  in3 = in3.random();

  float * gpu_in1_data  = static_cast<float*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(float)));
  float * gpu_in2_data  = static_cast<float*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(float)));
  float * gpu_in3_data  = static_cast<float*>(sycl_device.allocate(in3.dimensions().TotalSize()*sizeof(float)));
  float * gpu_out_data =  static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));

  TensorMap<Tensor<float, 3>> gpu_in1(gpu_in1_data, tensorRange);
  TensorMap<Tensor<float, 3>> gpu_in2(gpu_in2_data, tensorRange);
  TensorMap<Tensor<float, 3>> gpu_in3(gpu_in3_data, tensorRange);
  TensorMap<Tensor<float, 3>> gpu_out(gpu_out_data, tensorRange);

  /// a=1.2f
  gpu_in1.device(sycl_device) = gpu_in1.constant(1.2f);
  sycl_device.memcpyDeviceToHost(in1.data(), gpu_in1_data ,(in1.dimensions().TotalSize())*sizeof(float));
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(in1(i,j,k), 1.2f);
      }
    }
  }
  printf("a=1.2f Test passed\n");

  /// a=b*1.2f
  gpu_out.device(sycl_device) = gpu_in1 * 1.2f;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data ,(out.dimensions().TotalSize())*sizeof(float));
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) * 1.2f);
      }
    }
  }
  printf("a=b*1.2f Test Passed\n");

  /// c=a*b
  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.dimensions().TotalSize())*sizeof(float));
  gpu_out.device(sycl_device) = gpu_in1 * gpu_in2;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) *
                             in2(i,j,k));
      }
    }
  }
  printf("c=a*b Test Passed\n");

  /// c=a+b
  gpu_out.device(sycl_device) = gpu_in1 + gpu_in2;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) +
                             in2(i,j,k));
      }
    }
  }
  printf("c=a+b Test Passed\n");

  /// c=a*a
  gpu_out.device(sycl_device) = gpu_in1 * gpu_in1;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) *
                             in1(i,j,k));
      }
    }
  }
  printf("c= a*a Test Passed\n");

  //a*3.14f + b*2.7f
  gpu_out.device(sycl_device) =  gpu_in1 * gpu_in1.constant(3.14f) + gpu_in2 * gpu_in2.constant(2.7f);
  sycl_device.memcpyDeviceToHost(out.data(),gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) * 3.14f
                       + in2(i,j,k) * 2.7f);
      }
    }
  }
  printf("a*3.14f + b*2.7f Test Passed\n");

  ///d= (a>0.5? b:c)
  sycl_device.memcpyHostToDevice(gpu_in3_data, in3.data(),(in3.dimensions().TotalSize())*sizeof(float));
  gpu_out.device(sycl_device) =(gpu_in1 > gpu_in1.constant(0.5f)).select(gpu_in2, gpu_in3);
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i, j, k), (in1(i, j, k) > 0.5f)
                                                ? in2(i, j, k)
                                                : in3(i, j, k));
      }
    }
  }
  printf("d= (a>0.5? b:c) Test Passed\n");
  sycl_device.deallocate(gpu_in1_data);
  sycl_device.deallocate(gpu_in2_data);
  sycl_device.deallocate(gpu_in3_data);
  sycl_device.deallocate(gpu_out_data);
}
void test_cxx11_tensor_sycl() {
  cl::sycl::gpu_selector s;
  Eigen::SyclDevice sycl_device(s);
  CALL_SUBTEST(test_sycl_cpu(sycl_device));
}
