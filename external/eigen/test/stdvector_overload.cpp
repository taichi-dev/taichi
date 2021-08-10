// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/StdVector>
#include <Eigen/Geometry>

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Vector4f)

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Matrix2f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Matrix4f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Matrix4d)

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Affine3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Affine3d)

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Quaternionf)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Quaterniond)

template<typename MatrixType>
void check_stdvector_matrix(const MatrixType& m)
{
  typename MatrixType::Index rows = m.rows();
  typename MatrixType::Index cols = m.cols();
  MatrixType x = MatrixType::Random(rows,cols), y = MatrixType::Random(rows,cols);
  std::vector<MatrixType> v(10, MatrixType::Zero(rows,cols)), w(20, y);
  v[5] = x;
  w[6] = v[5];
  VERIFY_IS_APPROX(w[6], v[5]);
  v = w;
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(w[i], v[i]);
  }

  v.resize(21);
  v[20] = x;
  VERIFY_IS_APPROX(v[20], x);
  v.resize(22,y);
  VERIFY_IS_APPROX(v[21], y);
  v.push_back(x);
  VERIFY_IS_APPROX(v[22], x);
  VERIFY((internal::UIntPtr)&(v[22]) == (internal::UIntPtr)&(v[21]) + sizeof(MatrixType));

  // do a lot of push_back such that the vector gets internally resized
  // (with memory reallocation)
  MatrixType* ref = &w[0];
  for(int i=0; i<30 || ((ref==&w[0]) && i<300); ++i)
    v.push_back(w[i%w.size()]);
  for(unsigned int i=23; i<v.size(); ++i)
  {
    VERIFY(v[i]==w[(i-23)%w.size()]);
  }
}

template<typename TransformType>
void check_stdvector_transform(const TransformType&)
{
  typedef typename TransformType::MatrixType MatrixType;
  TransformType x(MatrixType::Random()), y(MatrixType::Random());
  std::vector<TransformType> v(10), w(20, y);
  v[5] = x;
  w[6] = v[5];
  VERIFY_IS_APPROX(w[6], v[5]);
  v = w;
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(w[i], v[i]);
  }

  v.resize(21);
  v[20] = x;
  VERIFY_IS_APPROX(v[20], x);
  v.resize(22,y);
  VERIFY_IS_APPROX(v[21], y);
  v.push_back(x);
  VERIFY_IS_APPROX(v[22], x);
  VERIFY((internal::UIntPtr)&(v[22]) == (internal::UIntPtr)&(v[21]) + sizeof(TransformType));

  // do a lot of push_back such that the vector gets internally resized
  // (with memory reallocation)
  TransformType* ref = &w[0];
  for(int i=0; i<30 || ((ref==&w[0]) && i<300); ++i)
    v.push_back(w[i%w.size()]);
  for(unsigned int i=23; i<v.size(); ++i)
  {
    VERIFY(v[i].matrix()==w[(i-23)%w.size()].matrix());
  }
}

template<typename QuaternionType>
void check_stdvector_quaternion(const QuaternionType&)
{
  typedef typename QuaternionType::Coefficients Coefficients;
  QuaternionType x(Coefficients::Random()), y(Coefficients::Random()), qi=QuaternionType::Identity();
  std::vector<QuaternionType> v(10,qi), w(20, y);
  v[5] = x;
  w[6] = v[5];
  VERIFY_IS_APPROX(w[6], v[5]);
  v = w;
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(w[i], v[i]);
  }

  v.resize(21);
  v[20] = x;
  VERIFY_IS_APPROX(v[20], x);
  v.resize(22,y);
  VERIFY_IS_APPROX(v[21], y);
  v.push_back(x);
  VERIFY_IS_APPROX(v[22], x);
  VERIFY((internal::UIntPtr)&(v[22]) == (internal::UIntPtr)&(v[21]) + sizeof(QuaternionType));

  // do a lot of push_back such that the vector gets internally resized
  // (with memory reallocation)
  QuaternionType* ref = &w[0];
  for(int i=0; i<30 || ((ref==&w[0]) && i<300); ++i)
    v.push_back(w[i%w.size()]);
  for(unsigned int i=23; i<v.size(); ++i)
  {
    VERIFY(v[i].coeffs()==w[(i-23)%w.size()].coeffs());
  }
}

void test_stdvector_overload()
{
  // some non vectorizable fixed sizes
  CALL_SUBTEST_1(check_stdvector_matrix(Vector2f()));
  CALL_SUBTEST_1(check_stdvector_matrix(Matrix3f()));
  CALL_SUBTEST_2(check_stdvector_matrix(Matrix3d()));

  // some vectorizable fixed sizes
  CALL_SUBTEST_1(check_stdvector_matrix(Matrix2f()));
  CALL_SUBTEST_1(check_stdvector_matrix(Vector4f()));
  CALL_SUBTEST_1(check_stdvector_matrix(Matrix4f()));
  CALL_SUBTEST_2(check_stdvector_matrix(Matrix4d()));

  // some dynamic sizes
  CALL_SUBTEST_3(check_stdvector_matrix(MatrixXd(1,1)));
  CALL_SUBTEST_3(check_stdvector_matrix(VectorXd(20)));
  CALL_SUBTEST_3(check_stdvector_matrix(RowVectorXf(20)));
  CALL_SUBTEST_3(check_stdvector_matrix(MatrixXcf(10,10)));

  // some Transform
  CALL_SUBTEST_4(check_stdvector_transform(Affine2f())); // does not need the specialization (2+1)^2 = 9
  CALL_SUBTEST_4(check_stdvector_transform(Affine3f()));
  CALL_SUBTEST_4(check_stdvector_transform(Affine3d()));

  // some Quaternion
  CALL_SUBTEST_5(check_stdvector_quaternion(Quaternionf()));
  CALL_SUBTEST_5(check_stdvector_quaternion(Quaterniond()));
}
