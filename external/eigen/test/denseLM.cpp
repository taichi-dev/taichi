// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <fstream>
#include <iomanip>

#include "main.h"
#include <Eigen/LevenbergMarquardt>
using namespace std;
using namespace Eigen;

template<typename Scalar>
struct DenseLM : DenseFunctor<Scalar>
{
  typedef DenseFunctor<Scalar> Base;
  typedef typename Base::JacobianType JacobianType;
  typedef Matrix<Scalar,Dynamic,1> VectorType;
  
  DenseLM(int n, int m) : DenseFunctor<Scalar>(n,m) 
  { }
 
  VectorType model(const VectorType& uv, VectorType& x)
  {
    VectorType y; // Should change to use expression template
    int m = Base::values(); 
    int n = Base::inputs();
    eigen_assert(uv.size()%2 == 0);
    eigen_assert(uv.size() == n);
    eigen_assert(x.size() == m);
    y.setZero(m);
    int half = n/2;
    VectorBlock<const VectorType> u(uv, 0, half);
    VectorBlock<const VectorType> v(uv, half, half);
    for (int j = 0; j < m; j++)
    {
      for (int i = 0; i < half; i++)
        y(j) += u(i)*std::exp(-(x(j)-i)*(x(j)-i)/(v(i)*v(i)));
    }
    return y;
    
  }
  void initPoints(VectorType& uv_ref, VectorType& x)
  {
    m_x = x;
    m_y = this->model(uv_ref, x);
  }
  
  int operator()(const VectorType& uv, VectorType& fvec)
  {
    
    int m = Base::values(); 
    int n = Base::inputs();
    eigen_assert(uv.size()%2 == 0);
    eigen_assert(uv.size() == n);
    eigen_assert(fvec.size() == m);
    int half = n/2;
    VectorBlock<const VectorType> u(uv, 0, half);
    VectorBlock<const VectorType> v(uv, half, half);
    for (int j = 0; j < m; j++)
    {
      fvec(j) = m_y(j);
      for (int i = 0; i < half; i++)
      {
        fvec(j) -= u(i) *std::exp(-(m_x(j)-i)*(m_x(j)-i)/(v(i)*v(i)));
      }
    }
    
    return 0;
  }
  int df(const VectorType& uv, JacobianType& fjac)
  {
    int m = Base::values(); 
    int n = Base::inputs();
    eigen_assert(n == uv.size());
    eigen_assert(fjac.rows() == m);
    eigen_assert(fjac.cols() == n);
    int half = n/2;
    VectorBlock<const VectorType> u(uv, 0, half);
    VectorBlock<const VectorType> v(uv, half, half);
    for (int j = 0; j < m; j++)
    {
      for (int i = 0; i < half; i++)
      {
        fjac.coeffRef(j,i) = -std::exp(-(m_x(j)-i)*(m_x(j)-i)/(v(i)*v(i)));
        fjac.coeffRef(j,i+half) = -2.*u(i)*(m_x(j)-i)*(m_x(j)-i)/(std::pow(v(i),3)) * std::exp(-(m_x(j)-i)*(m_x(j)-i)/(v(i)*v(i)));
      }
    }
    return 0;
  }
  VectorType m_x, m_y; //Data Points
};

template<typename FunctorType, typename VectorType>
int test_minimizeLM(FunctorType& functor, VectorType& uv)
{
  LevenbergMarquardt<FunctorType> lm(functor);
  LevenbergMarquardtSpace::Status info; 
  
  info = lm.minimize(uv);
  
  VERIFY_IS_EQUAL(info, 1);
  //FIXME Check other parameters
  return info;
}

template<typename FunctorType, typename VectorType>
int test_lmder(FunctorType& functor, VectorType& uv)
{
  typedef typename VectorType::Scalar Scalar;
  LevenbergMarquardtSpace::Status info; 
  LevenbergMarquardt<FunctorType> lm(functor);
  info = lm.lmder1(uv);
  
  VERIFY_IS_EQUAL(info, 1);
  //FIXME Check other parameters
  return info;
}

template<typename FunctorType, typename VectorType>
int test_minimizeSteps(FunctorType& functor, VectorType& uv)
{
  LevenbergMarquardtSpace::Status info;   
  LevenbergMarquardt<FunctorType> lm(functor);
  info = lm.minimizeInit(uv);
  if (info==LevenbergMarquardtSpace::ImproperInputParameters)
      return info;
  do 
  {
    info = lm.minimizeOneStep(uv);
  } while (info==LevenbergMarquardtSpace::Running);
  
  VERIFY_IS_EQUAL(info, 1);
  //FIXME Check other parameters
  return info;
}

template<typename T>
void test_denseLM_T()
{
  typedef Matrix<T,Dynamic,1> VectorType;
  
  int inputs = 10; 
  int values = 1000; 
  DenseLM<T> dense_gaussian(inputs, values);
  VectorType uv(inputs),uv_ref(inputs);
  VectorType x(values);
  
  // Generate the reference solution 
  uv_ref << -2, 1, 4 ,8, 6, 1.8, 1.2, 1.1, 1.9 , 3;
  
  //Generate the reference data points
  x.setRandom();
  x = 10*x;
  x.array() += 10;
  dense_gaussian.initPoints(uv_ref, x);
  
  // Generate the initial parameters 
  VectorBlock<VectorType> u(uv, 0, inputs/2); 
  VectorBlock<VectorType> v(uv, inputs/2, inputs/2);
  
  // Solve the optimization problem
  
  //Solve in one go
  u.setOnes(); v.setOnes();
  test_minimizeLM(dense_gaussian, uv);
  
  //Solve until the machine precision
  u.setOnes(); v.setOnes();
  test_lmder(dense_gaussian, uv); 
  
  // Solve step by step
  v.setOnes(); u.setOnes();
  test_minimizeSteps(dense_gaussian, uv);
  
}

void test_denseLM()
{
  CALL_SUBTEST_2(test_denseLM_T<double>());
  
  // CALL_SUBTEST_2(test_sparseLM_T<std::complex<double>());
}
