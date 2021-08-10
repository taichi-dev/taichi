// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define TEST_ENABLE_TEMPORARY_TRACKING
  
#include "main.h"

using namespace std;
template<typename MatrixType> void permutationmatrices(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime,
         Options = MatrixType::Options };
  typedef PermutationMatrix<Rows> LeftPermutationType;
  typedef Transpositions<Rows> LeftTranspositionsType;
  typedef Matrix<int, Rows, 1> LeftPermutationVectorType;
  typedef Map<LeftPermutationType> MapLeftPerm;
  typedef PermutationMatrix<Cols> RightPermutationType;
  typedef Transpositions<Cols> RightTranspositionsType;
  typedef Matrix<int, Cols, 1> RightPermutationVectorType;
  typedef Map<RightPermutationType> MapRightPerm;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m_original = MatrixType::Random(rows,cols);
  LeftPermutationVectorType lv;
  randomPermutationVector(lv, rows);
  LeftPermutationType lp(lv);
  RightPermutationVectorType rv;
  randomPermutationVector(rv, cols);
  RightPermutationType rp(rv);
  LeftTranspositionsType lt(lv);
  RightTranspositionsType rt(rv);
  MatrixType m_permuted = MatrixType::Random(rows,cols);
  
  VERIFY_EVALUATION_COUNT(m_permuted = lp * m_original * rp, 1); // 1 temp for sub expression "lp * m_original"

  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
        VERIFY_IS_APPROX(m_permuted(lv(i),j), m_original(i,rv(j)));

  Matrix<Scalar,Rows,Rows> lm(lp);
  Matrix<Scalar,Cols,Cols> rm(rp);

  VERIFY_IS_APPROX(m_permuted, lm*m_original*rm);
  
  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted = lp * m_permuted * rp, 1);
  VERIFY_IS_APPROX(m_permuted, lm*m_original*rm);
  
  VERIFY_IS_APPROX(lp.inverse()*m_permuted*rp.inverse(), m_original);
  VERIFY_IS_APPROX(lv.asPermutation().inverse()*m_permuted*rv.asPermutation().inverse(), m_original);
  VERIFY_IS_APPROX(MapLeftPerm(lv.data(),lv.size()).inverse()*m_permuted*MapRightPerm(rv.data(),rv.size()).inverse(), m_original);
  
  VERIFY((lp*lp.inverse()).toDenseMatrix().isIdentity());
  VERIFY((lv.asPermutation()*lv.asPermutation().inverse()).toDenseMatrix().isIdentity());
  VERIFY((MapLeftPerm(lv.data(),lv.size())*MapLeftPerm(lv.data(),lv.size()).inverse()).toDenseMatrix().isIdentity());

  LeftPermutationVectorType lv2;
  randomPermutationVector(lv2, rows);
  LeftPermutationType lp2(lv2);
  Matrix<Scalar,Rows,Rows> lm2(lp2);
  VERIFY_IS_APPROX((lp*lp2).toDenseMatrix().template cast<Scalar>(), lm*lm2);
  VERIFY_IS_APPROX((lv.asPermutation()*lv2.asPermutation()).toDenseMatrix().template cast<Scalar>(), lm*lm2);
  VERIFY_IS_APPROX((MapLeftPerm(lv.data(),lv.size())*MapLeftPerm(lv2.data(),lv2.size())).toDenseMatrix().template cast<Scalar>(), lm*lm2);

  LeftPermutationType identityp;
  identityp.setIdentity(rows);
  VERIFY_IS_APPROX(m_original, identityp*m_original);
  
  // check inplace permutations
  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias()= lp.inverse() * m_permuted, 1); // 1 temp to allocate the mask
  VERIFY_IS_APPROX(m_permuted, lp.inverse()*m_original);
  
  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = m_permuted * rp.inverse(), 1); // 1 temp to allocate the mask
  VERIFY_IS_APPROX(m_permuted, m_original*rp.inverse());
  
  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = lp * m_permuted, 1); // 1 temp to allocate the mask
  VERIFY_IS_APPROX(m_permuted, lp*m_original);
  
  m_permuted = m_original;
  VERIFY_EVALUATION_COUNT(m_permuted.noalias() = m_permuted * rp, 1); // 1 temp to allocate the mask
  VERIFY_IS_APPROX(m_permuted, m_original*rp);

  if(rows>1 && cols>1)
  {
    lp2 = lp;
    Index i = internal::random<Index>(0, rows-1);
    Index j;
    do j = internal::random<Index>(0, rows-1); while(j==i);
    lp2.applyTranspositionOnTheLeft(i, j);
    lm = lp;
    lm.row(i).swap(lm.row(j));
    VERIFY_IS_APPROX(lm, lp2.toDenseMatrix().template cast<Scalar>());

    RightPermutationType rp2 = rp;
    i = internal::random<Index>(0, cols-1);
    do j = internal::random<Index>(0, cols-1); while(j==i);
    rp2.applyTranspositionOnTheRight(i, j);
    rm = rp;
    rm.col(i).swap(rm.col(j));
    VERIFY_IS_APPROX(rm, rp2.toDenseMatrix().template cast<Scalar>());
  }

  {
    // simple compilation check
    Matrix<Scalar, Cols, Cols> A = rp;
    Matrix<Scalar, Cols, Cols> B = rp.transpose();
    VERIFY_IS_APPROX(A, B.transpose());
  }

  m_permuted = m_original;
  lp = lt;
  rp = rt;
  VERIFY_EVALUATION_COUNT(m_permuted = lt * m_permuted * rt, 1);
  VERIFY_IS_APPROX(m_permuted, lp*m_original*rp.transpose());
  
  VERIFY_IS_APPROX(lt.inverse()*m_permuted*rt.inverse(), m_original);
}

template<typename T>
void bug890()
{
  typedef Matrix<T, Dynamic, Dynamic> MatrixType;
  typedef Matrix<T, Dynamic, 1> VectorType;
  typedef Stride<Dynamic,Dynamic> S;
  typedef Map<MatrixType, Aligned, S> MapType;
  typedef PermutationMatrix<Dynamic> Perm;
  
  VectorType v1(2), v2(2), op(4), rhs(2);
  v1 << 666,667;
  op << 1,0,0,1;
  rhs << 42,42;
  
  Perm P(2);
  P.indices() << 1, 0;

  MapType(v1.data(),2,1,S(1,1)) = P * MapType(rhs.data(),2,1,S(1,1));
  VERIFY_IS_APPROX(v1, (P * rhs).eval());
  
  MapType(v1.data(),2,1,S(1,1)) = P.inverse() * MapType(rhs.data(),2,1,S(1,1));
  VERIFY_IS_APPROX(v1, (P.inverse() * rhs).eval());
}

void test_permutationmatrices()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( permutationmatrices(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( permutationmatrices(Matrix3f()) );
    CALL_SUBTEST_3( permutationmatrices(Matrix<double,3,3,RowMajor>()) );
    CALL_SUBTEST_4( permutationmatrices(Matrix4d()) );
    CALL_SUBTEST_5( permutationmatrices(Matrix<double,40,60>()) );
    CALL_SUBTEST_6( permutationmatrices(Matrix<double,Dynamic,Dynamic,RowMajor>(20, 30)) );
    CALL_SUBTEST_7( permutationmatrices(MatrixXcf(15, 10)) );
  }
  CALL_SUBTEST_5( bug890<double>() );
}
