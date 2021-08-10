// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_WORK_AROUND_QT_BUG_CALLING_WRONG_OPERATOR_NEW_FIXED_IN_QT_4_5

#include "main.h"
#include <QtCore/QVector>
#include <Eigen/Geometry>
#include <Eigen/QtAlignedMalloc>

template<typename MatrixType>
void check_qtvector_matrix(const MatrixType& m)
{
  Index rows = m.rows();
  Index cols = m.cols();
  MatrixType x = MatrixType::Random(rows,cols), y = MatrixType::Random(rows,cols);
  QVector<MatrixType> v(10, MatrixType(rows,cols)), w(20, y);
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(w[i], y);
  }
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
  v.fill(y,22);
  VERIFY_IS_APPROX(v[21], y);
  v.push_back(x);
  VERIFY_IS_APPROX(v[22], x);
  VERIFY((size_t)&(v[22]) == (size_t)&(v[21]) + sizeof(MatrixType));

  // do a lot of push_back such that the vector gets internally resized
  // (with memory reallocation)
  MatrixType* ref = &w[0];
  for(int i=0; i<30 || ((ref==&w[0]) && i<300); ++i)
    v.push_back(w[i%w.size()]);
  for(int i=23; i<v.size(); ++i)
  {
    VERIFY(v[i]==w[(i-23)%w.size()]);
  }
}

template<typename TransformType>
void check_qtvector_transform(const TransformType&)
{
  typedef typename TransformType::MatrixType MatrixType;
  TransformType x(MatrixType::Random()), y(MatrixType::Random());
  QVector<TransformType> v(10), w(20, y);
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
  v.fill(y,22);
  VERIFY_IS_APPROX(v[21], y);
  v.push_back(x);
  VERIFY_IS_APPROX(v[22], x);
  VERIFY((size_t)&(v[22]) == (size_t)&(v[21]) + sizeof(TransformType));

  // do a lot of push_back such that the vector gets internally resized
  // (with memory reallocation)
  TransformType* ref = &w[0];
  for(int i=0; i<30 || ((ref==&w[0]) && i<300); ++i)
    v.push_back(w[i%w.size()]);
  for(unsigned int i=23; int(i)<v.size(); ++i)
  {
    VERIFY(v[i].matrix()==w[(i-23)%w.size()].matrix());
  }
}

template<typename QuaternionType>
void check_qtvector_quaternion(const QuaternionType&)
{
  typedef typename QuaternionType::Coefficients Coefficients;
  QuaternionType x(Coefficients::Random()), y(Coefficients::Random());
  QVector<QuaternionType> v(10), w(20, y);
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
  v.fill(y,22);
  VERIFY_IS_APPROX(v[21], y);
  v.push_back(x);
  VERIFY_IS_APPROX(v[22], x);
  VERIFY((size_t)&(v[22]) == (size_t)&(v[21]) + sizeof(QuaternionType));

  // do a lot of push_back such that the vector gets internally resized
  // (with memory reallocation)
  QuaternionType* ref = &w[0];
  for(int i=0; i<30 || ((ref==&w[0]) && i<300); ++i)
    v.push_back(w[i%w.size()]);
  for(unsigned int i=23; int(i)<v.size(); ++i)
  {
    VERIFY(v[i].coeffs()==w[(i-23)%w.size()].coeffs());
  }
}

void test_qtvector()
{
  // some non vectorizable fixed sizes
  CALL_SUBTEST(check_qtvector_matrix(Vector2f()));
  CALL_SUBTEST(check_qtvector_matrix(Matrix3f()));
  CALL_SUBTEST(check_qtvector_matrix(Matrix3d()));

  // some vectorizable fixed sizes
  CALL_SUBTEST(check_qtvector_matrix(Matrix2f()));
  CALL_SUBTEST(check_qtvector_matrix(Vector4f()));
  CALL_SUBTEST(check_qtvector_matrix(Matrix4f()));
  CALL_SUBTEST(check_qtvector_matrix(Matrix4d()));

  // some dynamic sizes
  CALL_SUBTEST(check_qtvector_matrix(MatrixXd(1,1)));
  CALL_SUBTEST(check_qtvector_matrix(VectorXd(20)));
  CALL_SUBTEST(check_qtvector_matrix(RowVectorXf(20)));
  CALL_SUBTEST(check_qtvector_matrix(MatrixXcf(10,10)));

  // some Transform
  CALL_SUBTEST(check_qtvector_transform(Affine2f()));
  CALL_SUBTEST(check_qtvector_transform(Affine3f()));
  CALL_SUBTEST(check_qtvector_transform(Affine3d()));
  //CALL_SUBTEST(check_qtvector_transform(Transform4d()));

  // some Quaternion
  CALL_SUBTEST(check_qtvector_quaternion(Quaternionf()));
  CALL_SUBTEST(check_qtvector_quaternion(Quaternionf()));
}
