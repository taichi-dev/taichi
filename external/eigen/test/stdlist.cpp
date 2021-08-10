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
#include <Eigen/StdList>
#include <Eigen/Geometry>

template<typename MatrixType>
void check_stdlist_matrix(const MatrixType& m)
{
  Index rows = m.rows();
  Index cols = m.cols();
  MatrixType x = MatrixType::Random(rows,cols), y = MatrixType::Random(rows,cols);
  std::list<MatrixType,Eigen::aligned_allocator<MatrixType> > v(10, MatrixType::Zero(rows,cols)), w(20, y);
  v.front() = x;
  w.front() = w.back();
  VERIFY_IS_APPROX(w.front(), w.back());
  v = w;

  typename std::list<MatrixType,Eigen::aligned_allocator<MatrixType> >::iterator vi = v.begin();
  typename std::list<MatrixType,Eigen::aligned_allocator<MatrixType> >::iterator wi = w.begin();
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(*vi, *wi);
    ++vi;
    ++wi;
  }

  v.resize(21, MatrixType::Zero(rows,cols));  
  v.back() = x;
  VERIFY_IS_APPROX(v.back(), x);
  v.resize(22,y);
  VERIFY_IS_APPROX(v.back(), y);
  v.push_back(x);
  VERIFY_IS_APPROX(v.back(), x);
}

template<typename TransformType>
void check_stdlist_transform(const TransformType&)
{
  typedef typename TransformType::MatrixType MatrixType;
  TransformType x(MatrixType::Random()), y(MatrixType::Random()), ti=TransformType::Identity();
  std::list<TransformType,Eigen::aligned_allocator<TransformType> > v(10,ti), w(20, y);
  v.front() = x;
  w.front() = w.back();
  VERIFY_IS_APPROX(w.front(), w.back());
  v = w;

  typename std::list<TransformType,Eigen::aligned_allocator<TransformType> >::iterator vi = v.begin();
  typename std::list<TransformType,Eigen::aligned_allocator<TransformType> >::iterator wi = w.begin();
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(*vi, *wi);
    ++vi;
    ++wi;
  }

  v.resize(21, ti);
  v.back() = x;
  VERIFY_IS_APPROX(v.back(), x);
  v.resize(22,y);
  VERIFY_IS_APPROX(v.back(), y);
  v.push_back(x);
  VERIFY_IS_APPROX(v.back(), x);
}

template<typename QuaternionType>
void check_stdlist_quaternion(const QuaternionType&)
{
  typedef typename QuaternionType::Coefficients Coefficients;
  QuaternionType x(Coefficients::Random()), y(Coefficients::Random()), qi=QuaternionType::Identity();
  std::list<QuaternionType,Eigen::aligned_allocator<QuaternionType> > v(10,qi), w(20, y);
  v.front() = x;
  w.front() = w.back();
  VERIFY_IS_APPROX(w.front(), w.back());
  v = w;

  typename std::list<QuaternionType,Eigen::aligned_allocator<QuaternionType> >::iterator vi = v.begin();
  typename std::list<QuaternionType,Eigen::aligned_allocator<QuaternionType> >::iterator wi = w.begin();
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(*vi, *wi);
    ++vi;
    ++wi;
  }

  v.resize(21,qi);
  v.back() = x;
  VERIFY_IS_APPROX(v.back(), x);
  v.resize(22,y);
  VERIFY_IS_APPROX(v.back(), y);
  v.push_back(x);
  VERIFY_IS_APPROX(v.back(), x);
}

void test_stdlist()
{
  // some non vectorizable fixed sizes
  CALL_SUBTEST_1(check_stdlist_matrix(Vector2f()));
  CALL_SUBTEST_1(check_stdlist_matrix(Matrix3f()));
  CALL_SUBTEST_2(check_stdlist_matrix(Matrix3d()));

  // some vectorizable fixed sizes
  CALL_SUBTEST_1(check_stdlist_matrix(Matrix2f()));
  CALL_SUBTEST_1(check_stdlist_matrix(Vector4f()));
  CALL_SUBTEST_1(check_stdlist_matrix(Matrix4f()));
  CALL_SUBTEST_2(check_stdlist_matrix(Matrix4d()));

  // some dynamic sizes
  CALL_SUBTEST_3(check_stdlist_matrix(MatrixXd(1,1)));
  CALL_SUBTEST_3(check_stdlist_matrix(VectorXd(20)));
  CALL_SUBTEST_3(check_stdlist_matrix(RowVectorXf(20)));
  CALL_SUBTEST_3(check_stdlist_matrix(MatrixXcf(10,10)));

  // some Transform
  CALL_SUBTEST_4(check_stdlist_transform(Affine2f()));
  CALL_SUBTEST_4(check_stdlist_transform(Affine3f()));
  CALL_SUBTEST_4(check_stdlist_transform(Affine3d()));

  // some Quaternion
  CALL_SUBTEST_5(check_stdlist_quaternion(Quaternionf()));
  CALL_SUBTEST_5(check_stdlist_quaternion(Quaterniond()));
}
