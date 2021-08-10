// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include "svd_fill.h"
#include <limits>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>


template<typename MatrixType> void selfadjointeigensolver_essential_check(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  RealScalar eival_eps = numext::mini<RealScalar>(test_precision<RealScalar>(),  NumTraits<Scalar>::dummy_precision()*20000);
  
  SelfAdjointEigenSolver<MatrixType> eiSymm(m);
  VERIFY_IS_EQUAL(eiSymm.info(), Success);

  RealScalar scaling = m.cwiseAbs().maxCoeff();

  if(scaling<(std::numeric_limits<RealScalar>::min)())
  {
    VERIFY(eiSymm.eigenvalues().cwiseAbs().maxCoeff() <= (std::numeric_limits<RealScalar>::min)());
  }
  else
  {
    VERIFY_IS_APPROX((m.template selfadjointView<Lower>() * eiSymm.eigenvectors())/scaling,
                     (eiSymm.eigenvectors() * eiSymm.eigenvalues().asDiagonal())/scaling);
  }
  VERIFY_IS_APPROX(m.template selfadjointView<Lower>().eigenvalues(), eiSymm.eigenvalues());
  VERIFY_IS_UNITARY(eiSymm.eigenvectors());

  if(m.cols()<=4)
  {
    SelfAdjointEigenSolver<MatrixType> eiDirect;
    eiDirect.computeDirect(m);  
    VERIFY_IS_EQUAL(eiDirect.info(), Success);
    if(! eiSymm.eigenvalues().isApprox(eiDirect.eigenvalues(), eival_eps) )
    {
      std::cerr << "reference eigenvalues: " << eiSymm.eigenvalues().transpose() << "\n"
                << "obtained eigenvalues:  " << eiDirect.eigenvalues().transpose() << "\n"
                << "diff:                  " << (eiSymm.eigenvalues()-eiDirect.eigenvalues()).transpose() << "\n"
                << "error (eps):           " << (eiSymm.eigenvalues()-eiDirect.eigenvalues()).norm() / eiSymm.eigenvalues().norm() << "  (" << eival_eps << ")\n";
    }
    if(scaling<(std::numeric_limits<RealScalar>::min)())
    {
      VERIFY(eiDirect.eigenvalues().cwiseAbs().maxCoeff() <= (std::numeric_limits<RealScalar>::min)());
    }
    else
    {
      VERIFY_IS_APPROX(eiSymm.eigenvalues()/scaling, eiDirect.eigenvalues()/scaling);
      VERIFY_IS_APPROX((m.template selfadjointView<Lower>() * eiDirect.eigenvectors())/scaling,
                       (eiDirect.eigenvectors() * eiDirect.eigenvalues().asDiagonal())/scaling);
      VERIFY_IS_APPROX(m.template selfadjointView<Lower>().eigenvalues()/scaling, eiDirect.eigenvalues()/scaling);
    }

    VERIFY_IS_UNITARY(eiDirect.eigenvectors());
  }
}

template<typename MatrixType> void selfadjointeigensolver(const MatrixType& m)
{
  /* this test covers the following files:
     EigenSolver.h, SelfAdjointEigenSolver.h (and indirectly: Tridiagonalization.h)
  */
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  RealScalar largerEps = 10*test_precision<RealScalar>();

  MatrixType a = MatrixType::Random(rows,cols);
  MatrixType a1 = MatrixType::Random(rows,cols);
  MatrixType symmA =  a.adjoint() * a + a1.adjoint() * a1;
  MatrixType symmC = symmA;
  
  svd_fill_random(symmA,Symmetric);

  symmA.template triangularView<StrictlyUpper>().setZero();
  symmC.template triangularView<StrictlyUpper>().setZero();

  MatrixType b = MatrixType::Random(rows,cols);
  MatrixType b1 = MatrixType::Random(rows,cols);
  MatrixType symmB = b.adjoint() * b + b1.adjoint() * b1;
  symmB.template triangularView<StrictlyUpper>().setZero();
  
  CALL_SUBTEST( selfadjointeigensolver_essential_check(symmA) );

  SelfAdjointEigenSolver<MatrixType> eiSymm(symmA);
  // generalized eigen pb
  GeneralizedSelfAdjointEigenSolver<MatrixType> eiSymmGen(symmC, symmB);

  SelfAdjointEigenSolver<MatrixType> eiSymmNoEivecs(symmA, false);
  VERIFY_IS_EQUAL(eiSymmNoEivecs.info(), Success);
  VERIFY_IS_APPROX(eiSymm.eigenvalues(), eiSymmNoEivecs.eigenvalues());
  
  // generalized eigen problem Ax = lBx
  eiSymmGen.compute(symmC, symmB,Ax_lBx);
  VERIFY_IS_EQUAL(eiSymmGen.info(), Success);
  VERIFY((symmC.template selfadjointView<Lower>() * eiSymmGen.eigenvectors()).isApprox(
          symmB.template selfadjointView<Lower>() * (eiSymmGen.eigenvectors() * eiSymmGen.eigenvalues().asDiagonal()), largerEps));

  // generalized eigen problem BAx = lx
  eiSymmGen.compute(symmC, symmB,BAx_lx);
  VERIFY_IS_EQUAL(eiSymmGen.info(), Success);
  VERIFY((symmB.template selfadjointView<Lower>() * (symmC.template selfadjointView<Lower>() * eiSymmGen.eigenvectors())).isApprox(
         (eiSymmGen.eigenvectors() * eiSymmGen.eigenvalues().asDiagonal()), largerEps));

  // generalized eigen problem ABx = lx
  eiSymmGen.compute(symmC, symmB,ABx_lx);
  VERIFY_IS_EQUAL(eiSymmGen.info(), Success);
  VERIFY((symmC.template selfadjointView<Lower>() * (symmB.template selfadjointView<Lower>() * eiSymmGen.eigenvectors())).isApprox(
         (eiSymmGen.eigenvectors() * eiSymmGen.eigenvalues().asDiagonal()), largerEps));


  eiSymm.compute(symmC);
  MatrixType sqrtSymmA = eiSymm.operatorSqrt();
  VERIFY_IS_APPROX(MatrixType(symmC.template selfadjointView<Lower>()), sqrtSymmA*sqrtSymmA);
  VERIFY_IS_APPROX(sqrtSymmA, symmC.template selfadjointView<Lower>()*eiSymm.operatorInverseSqrt());

  MatrixType id = MatrixType::Identity(rows, cols);
  VERIFY_IS_APPROX(id.template selfadjointView<Lower>().operatorNorm(), RealScalar(1));

  SelfAdjointEigenSolver<MatrixType> eiSymmUninitialized;
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.info());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.eigenvalues());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.eigenvectors());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorSqrt());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorInverseSqrt());

  eiSymmUninitialized.compute(symmA, false);
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.eigenvectors());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorSqrt());
  VERIFY_RAISES_ASSERT(eiSymmUninitialized.operatorInverseSqrt());

  // test Tridiagonalization's methods
  Tridiagonalization<MatrixType> tridiag(symmC);
  VERIFY_IS_APPROX(tridiag.diagonal(), tridiag.matrixT().diagonal());
  VERIFY_IS_APPROX(tridiag.subDiagonal(), tridiag.matrixT().template diagonal<-1>());
  Matrix<RealScalar,Dynamic,Dynamic> T = tridiag.matrixT();
  if(rows>1 && cols>1) {
    // FIXME check that upper and lower part are 0:
    //VERIFY(T.topRightCorner(rows-2, cols-2).template triangularView<Upper>().isZero());
  }
  VERIFY_IS_APPROX(tridiag.diagonal(), T.diagonal());
  VERIFY_IS_APPROX(tridiag.subDiagonal(), T.template diagonal<1>());
  VERIFY_IS_APPROX(MatrixType(symmC.template selfadjointView<Lower>()), tridiag.matrixQ() * tridiag.matrixT().eval() * MatrixType(tridiag.matrixQ()).adjoint());
  VERIFY_IS_APPROX(MatrixType(symmC.template selfadjointView<Lower>()), tridiag.matrixQ() * tridiag.matrixT() * tridiag.matrixQ().adjoint());
  
  // Test computation of eigenvalues from tridiagonal matrix
  if(rows > 1)
  {
    SelfAdjointEigenSolver<MatrixType> eiSymmTridiag;
    eiSymmTridiag.computeFromTridiagonal(tridiag.matrixT().diagonal(), tridiag.matrixT().diagonal(-1), ComputeEigenvectors);
    VERIFY_IS_APPROX(eiSymm.eigenvalues(), eiSymmTridiag.eigenvalues());
    VERIFY_IS_APPROX(tridiag.matrixT(), eiSymmTridiag.eigenvectors().real() * eiSymmTridiag.eigenvalues().asDiagonal() * eiSymmTridiag.eigenvectors().real().transpose());
  }

  if (rows > 1 && rows < 20)
  {
    // Test matrix with NaN
    symmC(0,0) = std::numeric_limits<typename MatrixType::RealScalar>::quiet_NaN();
    SelfAdjointEigenSolver<MatrixType> eiSymmNaN(symmC);
    VERIFY_IS_EQUAL(eiSymmNaN.info(), NoConvergence);
  }

  // regression test for bug 1098
  {
    SelfAdjointEigenSolver<MatrixType> eig(a.adjoint() * a);
    eig.compute(a.adjoint() * a);
  }

  // regression test for bug 478
  {
    a.setZero();
    SelfAdjointEigenSolver<MatrixType> ei3(a);
    VERIFY_IS_EQUAL(ei3.info(), Success);
    VERIFY_IS_MUCH_SMALLER_THAN(ei3.eigenvalues().norm(),RealScalar(1));
    VERIFY((ei3.eigenvectors().transpose()*ei3.eigenvectors().transpose()).eval().isIdentity());
  }
}

template<int>
void bug_854()
{
  Matrix3d m;
  m << 850.961, 51.966, 0,
       51.966, 254.841, 0,
            0,       0, 0;
  selfadjointeigensolver_essential_check(m);
}

template<int>
void bug_1014()
{
  Matrix3d m;
  m <<        0.11111111111111114658, 0, 0,
       0,     0.11111111111111109107, 0,
       0, 0,  0.11111111111111107719;
  selfadjointeigensolver_essential_check(m);
}

template<int>
void bug_1225()
{
  Matrix3d m1, m2;
  m1.setRandom();
  m1 = m1*m1.transpose();
  m2 = m1.triangularView<Upper>();
  SelfAdjointEigenSolver<Matrix3d> eig1(m1);
  SelfAdjointEigenSolver<Matrix3d> eig2(m2.selfadjointView<Upper>());
  VERIFY_IS_APPROX(eig1.eigenvalues(), eig2.eigenvalues());
}

template<int>
void bug_1204()
{
  SparseMatrix<double> A(2,2);
  A.setIdentity();
  SelfAdjointEigenSolver<Eigen::SparseMatrix<double> > eig(A);
}

void test_eigensolver_selfadjoint()
{
  int s = 0;
  for(int i = 0; i < g_repeat; i++) {
    // trivial test for 1x1 matrices:
    CALL_SUBTEST_1( selfadjointeigensolver(Matrix<float, 1, 1>()));
    CALL_SUBTEST_1( selfadjointeigensolver(Matrix<double, 1, 1>()));
    // very important to test 3x3 and 2x2 matrices since we provide special paths for them
    CALL_SUBTEST_12( selfadjointeigensolver(Matrix2f()) );
    CALL_SUBTEST_12( selfadjointeigensolver(Matrix2d()) );
    CALL_SUBTEST_13( selfadjointeigensolver(Matrix3f()) );
    CALL_SUBTEST_13( selfadjointeigensolver(Matrix3d()) );
    CALL_SUBTEST_2( selfadjointeigensolver(Matrix4d()) );
    
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4);
    CALL_SUBTEST_3( selfadjointeigensolver(MatrixXf(s,s)) );
    CALL_SUBTEST_4( selfadjointeigensolver(MatrixXd(s,s)) );
    CALL_SUBTEST_5( selfadjointeigensolver(MatrixXcd(s,s)) );
    CALL_SUBTEST_9( selfadjointeigensolver(Matrix<std::complex<double>,Dynamic,Dynamic,RowMajor>(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)

    // some trivial but implementation-wise tricky cases
    CALL_SUBTEST_4( selfadjointeigensolver(MatrixXd(1,1)) );
    CALL_SUBTEST_4( selfadjointeigensolver(MatrixXd(2,2)) );
    CALL_SUBTEST_6( selfadjointeigensolver(Matrix<double,1,1>()) );
    CALL_SUBTEST_7( selfadjointeigensolver(Matrix<double,2,2>()) );
  }
  
  CALL_SUBTEST_13( bug_854<0>() );
  CALL_SUBTEST_13( bug_1014<0>() );
  CALL_SUBTEST_13( bug_1204<0>() );
  CALL_SUBTEST_13( bug_1225<0>() );

  // Test problem size constructors
  s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4);
  CALL_SUBTEST_8(SelfAdjointEigenSolver<MatrixXf> tmp1(s));
  CALL_SUBTEST_8(Tridiagonalization<MatrixXf> tmp2(s));
  
  TEST_SET_BUT_UNUSED_VARIABLE(s)
}

