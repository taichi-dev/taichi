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

EIGEN_DEFINE_STL_LIST_SPECIALIZATION(Vector4f)

EIGEN_DEFINE_STL_LIST_SPECIALIZATION(Matrix2f)
EIGEN_DEFINE_STL_LIST_SPECIALIZATION(Matrix4f)
EIGEN_DEFINE_STL_LIST_SPECIALIZATION(Matrix4d)

EIGEN_DEFINE_STL_LIST_SPECIALIZATION(Affine3f)
EIGEN_DEFINE_STL_LIST_SPECIALIZATION(Affine3d)

EIGEN_DEFINE_STL_LIST_SPECIALIZATION(Quaternionf)
EIGEN_DEFINE_STL_LIST_SPECIALIZATION(Quaterniond)

template <class Container, class Position>
typename Container::iterator get(Container & c, Position position)
{
  typename Container::iterator it = c.begin();
  std::advance(it, position);
  return it;
}

template <class Container, class Position, class Value>
void set(Container & c, Position position, const Value & value)
{
  typename Container::iterator it = c.begin();
  std::advance(it, position);
  *it = value;
}

template<typename MatrixType>
void check_stdlist_matrix(const MatrixType& m)
{
  typename MatrixType::Index rows = m.rows();
  typename MatrixType::Index cols = m.cols();
  MatrixType x = MatrixType::Random(rows,cols), y = MatrixType::Random(rows,cols);
  std::list<MatrixType> v(10, MatrixType::Zero(rows,cols)), w(20, y);
  typename std::list<MatrixType>::iterator itv = get(v, 5);
  typename std::list<MatrixType>::iterator itw = get(w, 6);
  *itv = x;
  *itw = *itv;
  VERIFY_IS_APPROX(*itw, *itv);
  v = w;
  itv = v.begin();
  itw = w.begin();
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(*itw, *itv);
    ++itv;
    ++itw;
  }

  v.resize(21);
  set(v, 20, x);
  VERIFY_IS_APPROX(*get(v, 20), x);
  v.resize(22,y);
  VERIFY_IS_APPROX(*get(v, 21), y);
  v.push_back(x);
  VERIFY_IS_APPROX(*get(v, 22), x);

  // do a lot of push_back such that the list gets internally resized
  // (with memory reallocation)
  MatrixType* ref = &(*get(w, 0));
  for(int i=0; i<30 || ((ref==&(*get(w, 0))) && i<300); ++i)
    v.push_back(*get(w, i%w.size()));
  for(unsigned int i=23; i<v.size(); ++i)
  {
    VERIFY((*get(v, i))==(*get(w, (i-23)%w.size())));
  }
}

template<typename TransformType>
void check_stdlist_transform(const TransformType&)
{
  typedef typename TransformType::MatrixType MatrixType;
  TransformType x(MatrixType::Random()), y(MatrixType::Random()), ti=TransformType::Identity();
  std::list<TransformType> v(10,ti), w(20, y);
  typename std::list<TransformType>::iterator itv = get(v, 5);
  typename std::list<TransformType>::iterator itw = get(w, 6);
  *itv = x;
  *itw = *itv;
  VERIFY_IS_APPROX(*itw, *itv);
  v = w;
  itv = v.begin();
  itw = w.begin();
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(*itw, *itv);
    ++itv;
    ++itw;
  }

  v.resize(21, ti);
  set(v, 20, x);
  VERIFY_IS_APPROX(*get(v, 20), x);
  v.resize(22,y);
  VERIFY_IS_APPROX(*get(v, 21), y);
  v.push_back(x);
  VERIFY_IS_APPROX(*get(v, 22), x);

  // do a lot of push_back such that the list gets internally resized
  // (with memory reallocation)
  TransformType* ref = &(*get(w, 0));
  for(int i=0; i<30 || ((ref==&(*get(w, 0))) && i<300); ++i)
    v.push_back(*get(w, i%w.size()));
  for(unsigned int i=23; i<v.size(); ++i)
  {
    VERIFY(get(v, i)->matrix()==get(w, (i-23)%w.size())->matrix());
  }
}

template<typename QuaternionType>
void check_stdlist_quaternion(const QuaternionType&)
{
  typedef typename QuaternionType::Coefficients Coefficients;
  QuaternionType x(Coefficients::Random()), y(Coefficients::Random()), qi=QuaternionType::Identity();
  std::list<QuaternionType> v(10,qi), w(20, y);
  typename std::list<QuaternionType>::iterator itv = get(v, 5);
  typename std::list<QuaternionType>::iterator itw = get(w, 6);
  *itv = x;
  *itw = *itv;
  VERIFY_IS_APPROX(*itw, *itv);
  v = w;
  itv = v.begin();
  itw = w.begin();
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(*itw, *itv);
    ++itv;
    ++itw;
  }

  v.resize(21,qi);
  set(v, 20, x);
  VERIFY_IS_APPROX(*get(v, 20), x);
  v.resize(22,y);
  VERIFY_IS_APPROX(*get(v, 21), y);
  v.push_back(x);
  VERIFY_IS_APPROX(*get(v, 22), x);

  // do a lot of push_back such that the list gets internally resized
  // (with memory reallocation)
  QuaternionType* ref = &(*get(w, 0));
  for(int i=0; i<30 || ((ref==&(*get(w, 0))) && i<300); ++i)
    v.push_back(*get(w, i%w.size()));
  for(unsigned int i=23; i<v.size(); ++i)
  {
    VERIFY(get(v, i)->coeffs()==get(w, (i-23)%w.size())->coeffs());
  }
}

void test_stdlist_overload()
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
  CALL_SUBTEST_4(check_stdlist_transform(Affine2f())); // does not need the specialization (2+1)^2 = 9
  CALL_SUBTEST_4(check_stdlist_transform(Affine3f()));
  CALL_SUBTEST_4(check_stdlist_transform(Affine3d()));

  // some Quaternion
  CALL_SUBTEST_5(check_stdlist_quaternion(Quaternionf()));
  CALL_SUBTEST_5(check_stdlist_quaternion(Quaterniond()));
}
