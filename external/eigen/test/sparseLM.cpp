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

template <typename Scalar>
struct sparseGaussianTest : SparseFunctor<Scalar, int>
{
  typedef Matrix<Scalar,Dynamic,1> VectorType;
  typedef SparseFunctor<Scalar,int> Base;
  typedef typename Base::JacobianType JacobianType;
  sparseGaussianTest(int inputs, int values) : SparseFunctor<Scalar,int>(inputs,values)
  { }
  
  VectorType model(const VectorType& uv, VectorType& x)
  {
    VectorType y; //Change this to use expression template
    int m = Base::values(); 
    int n = Base::inputs();
    eigen_assert(uv.size()%2 == 0);
    eigen_assert(uv.size() == n);
    eigen_assert(x.size() == m);
    y.setZero(m);
    int half = n/2;
    VectorBlock<const VectorType> u(uv, 0, half);
    VectorBlock<const VectorType> v(uv, half, half);
    Scalar coeff;
    for (int j = 0; j < m; j++)
    {
      for (int i = 0; i < half; i++) 
      {
        coeff = (x(j)-i)/v(i);
        coeff *= coeff;
        if (coeff < 1. && coeff > 0.)
          y(j) += u(i)*std::pow((1-coeff), 2);
      }
    }
    return y;
  }
  void initPoints(VectorType& uv_ref, VectorType& x)
  {
    m_x = x;
    m_y = this->model(uv_ref,x);
  }
  int operator()(const VectorType& uv, VectorType& fvec)
  {
    int m = Base::values(); 
    int n = Base::inputs();
    eigen_assert(uv.size()%2 == 0);
    eigen_assert(uv.size() == n);
    int half = n/2;
    VectorBlock<const VectorType> u(uv, 0, half);
    VectorBlock<const VectorType> v(uv, half, half);
    fvec = m_y;
    Scalar coeff;
    for (int j = 0; j < m; j++)
    {
      for (int i = 0; i < half; i++)
      {
        coeff = (m_x(j)-i)/v(i);
        coeff *= coeff;
        if (coeff < 1. && coeff > 0.)
          fvec(j) -= u(i)*std::pow((1-coeff), 2);
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
    Scalar coeff;
    
    //Derivatives with respect to u
    for (int col = 0; col < half; col++)
    {
      for (int row = 0; row < m; row++)
      {
        coeff = (m_x(row)-col)/v(col);
          coeff = coeff*coeff;
        if(coeff < 1. && coeff > 0.)
        {
          fjac.coeffRef(row,col) = -(1-coeff)*(1-coeff);
        }
      }
    }
    //Derivatives with respect to v
    for (int col = 0; col < half; col++)
    {
      for (int row = 0; row < m; row++)
      {
        coeff = (m_x(row)-col)/v(col);
        coeff = coeff*coeff;
        if(coeff < 1. && coeff > 0.)
        {
          fjac.coeffRef(row,col+half) = -4 * (u(col)/v(col))*coeff*(1-coeff);
        }
      }
    }
    return 0;
  }
  
  VectorType m_x, m_y; //Data points
};


template<typename T>
void test_sparseLM_T()
{
  typedef Matrix<T,Dynamic,1> VectorType;
  
  int inputs = 10;
  int values = 2000;
  sparseGaussianTest<T> sparse_gaussian(inputs, values);
  VectorType uv(inputs),uv_ref(inputs);
  VectorType x(values);
  // Generate the reference solution 
  uv_ref << -2, 1, 4 ,8, 6, 1.8, 1.2, 1.1, 1.9 , 3;
  //Generate the reference data points
  x.setRandom();
  x = 10*x;
  x.array() += 10;
  sparse_gaussian.initPoints(uv_ref, x);
  
  
  // Generate the initial parameters 
  VectorBlock<VectorType> u(uv, 0, inputs/2); 
  VectorBlock<VectorType> v(uv, inputs/2, inputs/2);
  v.setOnes();
  //Generate u or Solve for u from v
  u.setOnes();
  
  // Solve the optimization problem
  LevenbergMarquardt<sparseGaussianTest<T> > lm(sparse_gaussian);
  int info;
//   info = lm.minimize(uv);
  
  VERIFY_IS_EQUAL(info,1);
    // Do a step by step solution and save the residual 
  int maxiter = 200;
  int iter = 0;
  MatrixXd Err(values, maxiter);
  MatrixXd Mod(values, maxiter);
  LevenbergMarquardtSpace::Status status; 
  status = lm.minimizeInit(uv);
  if (status==LevenbergMarquardtSpace::ImproperInputParameters)
      return ;

}
void test_sparseLM()
{
  CALL_SUBTEST_1(test_sparseLM_T<double>());
  
  // CALL_SUBTEST_2(test_sparseLM_T<std::complex<double>());
}
