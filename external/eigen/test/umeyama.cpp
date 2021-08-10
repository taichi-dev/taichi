// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <Eigen/LU> // required for MatrixBase::determinant
#include <Eigen/SVD> // required for SVD

using namespace Eigen;

//  Constructs a random matrix from the unitary group U(size).
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> randMatrixUnitary(int size)
{
  typedef T Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  MatrixType Q;

  int max_tries = 40;
  double is_unitary = false;

  while (!is_unitary && max_tries > 0)
  {
    // initialize random matrix
    Q = MatrixType::Random(size, size);

    // orthogonalize columns using the Gram-Schmidt algorithm
    for (int col = 0; col < size; ++col)
    {
      typename MatrixType::ColXpr colVec = Q.col(col);
      for (int prevCol = 0; prevCol < col; ++prevCol)
      {
        typename MatrixType::ColXpr prevColVec = Q.col(prevCol);
        colVec -= colVec.dot(prevColVec)*prevColVec;
      }
      Q.col(col) = colVec.normalized();
    }

    // this additional orthogonalization is not necessary in theory but should enhance
    // the numerical orthogonality of the matrix
    for (int row = 0; row < size; ++row)
    {
      typename MatrixType::RowXpr rowVec = Q.row(row);
      for (int prevRow = 0; prevRow < row; ++prevRow)
      {
        typename MatrixType::RowXpr prevRowVec = Q.row(prevRow);
        rowVec -= rowVec.dot(prevRowVec)*prevRowVec;
      }
      Q.row(row) = rowVec.normalized();
    }

    // final check
    is_unitary = Q.isUnitary();
    --max_tries;
  }

  if (max_tries == 0)
    eigen_assert(false && "randMatrixUnitary: Could not construct unitary matrix!");

  return Q;
}

//  Constructs a random matrix from the special unitary group SU(size).
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> randMatrixSpecialUnitary(int size)
{
  typedef T Scalar;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  // initialize unitary matrix
  MatrixType Q = randMatrixUnitary<Scalar>(size);

  // tweak the first column to make the determinant be 1
  Q.col(0) *= numext::conj(Q.determinant());

  return Q;
}

template <typename MatrixType>
void run_test(int dim, int num_elements)
{
  using std::abs;
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  typedef Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
  typedef Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

  // MUST be positive because in any other case det(cR_t) may become negative for
  // odd dimensions!
  const Scalar c = abs(internal::random<Scalar>());

  MatrixX R = randMatrixSpecialUnitary<Scalar>(dim);
  VectorX t = Scalar(50)*VectorX::Random(dim,1);

  MatrixX cR_t = MatrixX::Identity(dim+1,dim+1);
  cR_t.block(0,0,dim,dim) = c*R;
  cR_t.block(0,dim,dim,1) = t;

  MatrixX src = MatrixX::Random(dim+1, num_elements);
  src.row(dim) = Matrix<Scalar, 1, Dynamic>::Constant(num_elements, Scalar(1));

  MatrixX dst = cR_t*src;

  MatrixX cR_t_umeyama = umeyama(src.block(0,0,dim,num_elements), dst.block(0,0,dim,num_elements));

  const Scalar error = ( cR_t_umeyama*src - dst ).norm() / dst.norm();
  VERIFY(error < Scalar(40)*std::numeric_limits<Scalar>::epsilon());
}

template<typename Scalar, int Dimension>
void run_fixed_size_test(int num_elements)
{
  using std::abs;
  typedef Matrix<Scalar, Dimension+1, Dynamic> MatrixX;
  typedef Matrix<Scalar, Dimension+1, Dimension+1> HomMatrix;
  typedef Matrix<Scalar, Dimension, Dimension> FixedMatrix;
  typedef Matrix<Scalar, Dimension, 1> FixedVector;

  const int dim = Dimension;

  // MUST be positive because in any other case det(cR_t) may become negative for
  // odd dimensions!
  // Also if c is to small compared to t.norm(), problem is ill-posed (cf. Bug 744)
  const Scalar c = internal::random<Scalar>(0.5, 2.0);

  FixedMatrix R = randMatrixSpecialUnitary<Scalar>(dim);
  FixedVector t = Scalar(32)*FixedVector::Random(dim,1);

  HomMatrix cR_t = HomMatrix::Identity(dim+1,dim+1);
  cR_t.block(0,0,dim,dim) = c*R;
  cR_t.block(0,dim,dim,1) = t;

  MatrixX src = MatrixX::Random(dim+1, num_elements);
  src.row(dim) = Matrix<Scalar, 1, Dynamic>::Constant(num_elements, Scalar(1));

  MatrixX dst = cR_t*src;

  Block<MatrixX, Dimension, Dynamic> src_block(src,0,0,dim,num_elements);
  Block<MatrixX, Dimension, Dynamic> dst_block(dst,0,0,dim,num_elements);

  HomMatrix cR_t_umeyama = umeyama(src_block, dst_block);

  const Scalar error = ( cR_t_umeyama*src - dst ).squaredNorm();

  VERIFY(error < Scalar(16)*std::numeric_limits<Scalar>::epsilon());
}

void test_umeyama()
{
  for (int i=0; i<g_repeat; ++i)
  {
    const int num_elements = internal::random<int>(40,500);

    // works also for dimensions bigger than 3...
    for (int dim=2; dim<8; ++dim)
    {
      CALL_SUBTEST_1(run_test<MatrixXd>(dim, num_elements));
      CALL_SUBTEST_2(run_test<MatrixXf>(dim, num_elements));
    }

    CALL_SUBTEST_3((run_fixed_size_test<float, 2>(num_elements)));
    CALL_SUBTEST_4((run_fixed_size_test<float, 3>(num_elements)));
    CALL_SUBTEST_5((run_fixed_size_test<float, 4>(num_elements)));

    CALL_SUBTEST_6((run_fixed_size_test<double, 2>(num_elements)));
    CALL_SUBTEST_7((run_fixed_size_test<double, 3>(num_elements)));
    CALL_SUBTEST_8((run_fixed_size_test<double, 4>(num_elements)));
  }

  // Those two calls don't compile and result in meaningful error messages!
  // umeyama(MatrixXcf(),MatrixXcf());
  // umeyama(MatrixXcd(),MatrixXcd());
}
