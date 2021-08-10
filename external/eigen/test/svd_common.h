// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SVD_DEFAULT
#error a macro SVD_DEFAULT(MatrixType) must be defined prior to including svd_common.h
#endif

#ifndef SVD_FOR_MIN_NORM
#error a macro SVD_FOR_MIN_NORM(MatrixType) must be defined prior to including svd_common.h
#endif

#include "svd_fill.h"

// Check that the matrix m is properly reconstructed and that the U and V factors are unitary
// The SVD must have already been computed.
template<typename SvdType, typename MatrixType>
void svd_check_full(const MatrixType& m, const SvdType& svd)
{
  Index rows = m.rows();
  Index cols = m.cols();

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> MatrixUType;
  typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime> MatrixVType;

  MatrixType sigma = MatrixType::Zero(rows,cols);
  sigma.diagonal() = svd.singularValues().template cast<Scalar>();
  MatrixUType u = svd.matrixU();
  MatrixVType v = svd.matrixV();
  RealScalar scaling = m.cwiseAbs().maxCoeff();
  if(scaling<(std::numeric_limits<RealScalar>::min)())
  {
    VERIFY(sigma.cwiseAbs().maxCoeff() <= (std::numeric_limits<RealScalar>::min)());
  }
  else
  {
    VERIFY_IS_APPROX(m/scaling, u * (sigma/scaling) * v.adjoint());
  }
  VERIFY_IS_UNITARY(u);
  VERIFY_IS_UNITARY(v);
}

// Compare partial SVD defined by computationOptions to a full SVD referenceSvd
template<typename SvdType, typename MatrixType>
void svd_compare_to_full(const MatrixType& m,
                         unsigned int computationOptions,
                         const SvdType& referenceSvd)
{
  typedef typename MatrixType::RealScalar RealScalar;
  Index rows = m.rows();
  Index cols = m.cols();
  Index diagSize = (std::min)(rows, cols);
  RealScalar prec = test_precision<RealScalar>();

  SvdType svd(m, computationOptions);

  VERIFY_IS_APPROX(svd.singularValues(), referenceSvd.singularValues());
  
  if(computationOptions & (ComputeFullV|ComputeThinV))
  {
    VERIFY( (svd.matrixV().adjoint()*svd.matrixV()).isIdentity(prec) );
    VERIFY_IS_APPROX( svd.matrixV().leftCols(diagSize) * svd.singularValues().asDiagonal() * svd.matrixV().leftCols(diagSize).adjoint(),
                      referenceSvd.matrixV().leftCols(diagSize) * referenceSvd.singularValues().asDiagonal() * referenceSvd.matrixV().leftCols(diagSize).adjoint());
  }
  
  if(computationOptions & (ComputeFullU|ComputeThinU))
  {
    VERIFY( (svd.matrixU().adjoint()*svd.matrixU()).isIdentity(prec) );
    VERIFY_IS_APPROX( svd.matrixU().leftCols(diagSize) * svd.singularValues().cwiseAbs2().asDiagonal() * svd.matrixU().leftCols(diagSize).adjoint(),
                      referenceSvd.matrixU().leftCols(diagSize) * referenceSvd.singularValues().cwiseAbs2().asDiagonal() * referenceSvd.matrixU().leftCols(diagSize).adjoint());
  }
  
  // The following checks are not critical.
  // For instance, with Dived&Conquer SVD, if only the factor 'V' is computedt then different matrix-matrix product implementation will be used
  // and the resulting 'V' factor might be significantly different when the SVD decomposition is not unique, especially with single precision float.
  ++g_test_level;
  if(computationOptions & ComputeFullU)  VERIFY_IS_APPROX(svd.matrixU(), referenceSvd.matrixU());
  if(computationOptions & ComputeThinU)  VERIFY_IS_APPROX(svd.matrixU(), referenceSvd.matrixU().leftCols(diagSize));
  if(computationOptions & ComputeFullV)  VERIFY_IS_APPROX(svd.matrixV().cwiseAbs(), referenceSvd.matrixV().cwiseAbs());
  if(computationOptions & ComputeThinV)  VERIFY_IS_APPROX(svd.matrixV(), referenceSvd.matrixV().leftCols(diagSize));
  --g_test_level;
}

//
template<typename SvdType, typename MatrixType>
void svd_least_square(const MatrixType& m, unsigned int computationOptions)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  Index rows = m.rows();
  Index cols = m.cols();

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };

  typedef Matrix<Scalar, RowsAtCompileTime, Dynamic> RhsType;
  typedef Matrix<Scalar, ColsAtCompileTime, Dynamic> SolutionType;

  RhsType rhs = RhsType::Random(rows, internal::random<Index>(1, cols));
  SvdType svd(m, computationOptions);

       if(internal::is_same<RealScalar,double>::value) svd.setThreshold(1e-8);
  else if(internal::is_same<RealScalar,float>::value)  svd.setThreshold(2e-4);

  SolutionType x = svd.solve(rhs);
   
  RealScalar residual = (m*x-rhs).norm();
  RealScalar rhs_norm = rhs.norm();
  if(!test_isMuchSmallerThan(residual,rhs.norm()))
  {
    // ^^^ If the residual is very small, then we have an exact solution, so we are already good.
    
    // evaluate normal equation which works also for least-squares solutions
    if(internal::is_same<RealScalar,double>::value || svd.rank()==m.diagonal().size())
    {
      using std::sqrt;
      // This test is not stable with single precision.
      // This is probably because squaring m signicantly affects the precision.      
      if(internal::is_same<RealScalar,float>::value) ++g_test_level;
      
      VERIFY_IS_APPROX(m.adjoint()*(m*x),m.adjoint()*rhs);
      
      if(internal::is_same<RealScalar,float>::value) --g_test_level;
    }
    
    // Check that there is no significantly better solution in the neighborhood of x
    for(Index k=0;k<x.rows();++k)
    {
      using std::abs;
      
      SolutionType y(x);
      y.row(k) = (RealScalar(1)+2*NumTraits<RealScalar>::epsilon())*x.row(k);
      RealScalar residual_y = (m*y-rhs).norm();
      VERIFY( test_isMuchSmallerThan(abs(residual_y-residual), rhs_norm) || residual < residual_y );
      if(internal::is_same<RealScalar,float>::value) ++g_test_level;
      VERIFY( test_isApprox(residual_y,residual) || residual < residual_y );
      if(internal::is_same<RealScalar,float>::value) --g_test_level;
      
      y.row(k) = (RealScalar(1)-2*NumTraits<RealScalar>::epsilon())*x.row(k);
      residual_y = (m*y-rhs).norm();
      VERIFY( test_isMuchSmallerThan(abs(residual_y-residual), rhs_norm) || residual < residual_y );
      if(internal::is_same<RealScalar,float>::value) ++g_test_level;
      VERIFY( test_isApprox(residual_y,residual) || residual < residual_y );
      if(internal::is_same<RealScalar,float>::value) --g_test_level;
    }
  }
}

// check minimal norm solutions, the inoput matrix m is only used to recover problem size
template<typename MatrixType>
void svd_min_norm(const MatrixType& m, unsigned int computationOptions)
{
  typedef typename MatrixType::Scalar Scalar;
  Index cols = m.cols();

  enum {
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };

  typedef Matrix<Scalar, ColsAtCompileTime, Dynamic> SolutionType;

  // generate a full-rank m x n problem with m<n
  enum {
    RankAtCompileTime2 = ColsAtCompileTime==Dynamic ? Dynamic : (ColsAtCompileTime)/2+1,
    RowsAtCompileTime3 = ColsAtCompileTime==Dynamic ? Dynamic : ColsAtCompileTime+1
  };
  typedef Matrix<Scalar, RankAtCompileTime2, ColsAtCompileTime> MatrixType2;
  typedef Matrix<Scalar, RankAtCompileTime2, 1> RhsType2;
  typedef Matrix<Scalar, ColsAtCompileTime, RankAtCompileTime2> MatrixType2T;
  Index rank = RankAtCompileTime2==Dynamic ? internal::random<Index>(1,cols) : Index(RankAtCompileTime2);
  MatrixType2 m2(rank,cols);
  int guard = 0;
  do {
    m2.setRandom();
  } while(SVD_FOR_MIN_NORM(MatrixType2)(m2).setThreshold(test_precision<Scalar>()).rank()!=rank && (++guard)<10);
  VERIFY(guard<10);

  RhsType2 rhs2 = RhsType2::Random(rank);
  // use QR to find a reference minimal norm solution
  HouseholderQR<MatrixType2T> qr(m2.adjoint());
  Matrix<Scalar,Dynamic,1> tmp = qr.matrixQR().topLeftCorner(rank,rank).template triangularView<Upper>().adjoint().solve(rhs2);
  tmp.conservativeResize(cols);
  tmp.tail(cols-rank).setZero();
  SolutionType x21 = qr.householderQ() * tmp;
  // now check with SVD
  SVD_FOR_MIN_NORM(MatrixType2) svd2(m2, computationOptions);
  SolutionType x22 = svd2.solve(rhs2);
  VERIFY_IS_APPROX(m2*x21, rhs2);
  VERIFY_IS_APPROX(m2*x22, rhs2);
  VERIFY_IS_APPROX(x21, x22);

  // Now check with a rank deficient matrix
  typedef Matrix<Scalar, RowsAtCompileTime3, ColsAtCompileTime> MatrixType3;
  typedef Matrix<Scalar, RowsAtCompileTime3, 1> RhsType3;
  Index rows3 = RowsAtCompileTime3==Dynamic ? internal::random<Index>(rank+1,2*cols) : Index(RowsAtCompileTime3);
  Matrix<Scalar,RowsAtCompileTime3,Dynamic> C = Matrix<Scalar,RowsAtCompileTime3,Dynamic>::Random(rows3,rank);
  MatrixType3 m3 = C * m2;
  RhsType3 rhs3 = C * rhs2;
  SVD_FOR_MIN_NORM(MatrixType3) svd3(m3, computationOptions);
  SolutionType x3 = svd3.solve(rhs3);
  VERIFY_IS_APPROX(m3*x3, rhs3);
  VERIFY_IS_APPROX(m3*x21, rhs3);
  VERIFY_IS_APPROX(m2*x3, rhs2);
  VERIFY_IS_APPROX(x21, x3);
}

// Check full, compare_to_full, least_square, and min_norm for all possible compute-options
template<typename SvdType, typename MatrixType>
void svd_test_all_computation_options(const MatrixType& m, bool full_only)
{
//   if (QRPreconditioner == NoQRPreconditioner && m.rows() != m.cols())
//     return;
  SvdType fullSvd(m, ComputeFullU|ComputeFullV);
  CALL_SUBTEST(( svd_check_full(m, fullSvd) ));
  CALL_SUBTEST(( svd_least_square<SvdType>(m, ComputeFullU | ComputeFullV) ));
  CALL_SUBTEST(( svd_min_norm(m, ComputeFullU | ComputeFullV) ));
  
  #if defined __INTEL_COMPILER
  // remark #111: statement is unreachable
  #pragma warning disable 111
  #endif
  if(full_only)
    return;

  CALL_SUBTEST(( svd_compare_to_full(m, ComputeFullU, fullSvd) ));
  CALL_SUBTEST(( svd_compare_to_full(m, ComputeFullV, fullSvd) ));
  CALL_SUBTEST(( svd_compare_to_full(m, 0, fullSvd) ));

  if (MatrixType::ColsAtCompileTime == Dynamic) {
    // thin U/V are only available with dynamic number of columns
    CALL_SUBTEST(( svd_compare_to_full(m, ComputeFullU|ComputeThinV, fullSvd) ));
    CALL_SUBTEST(( svd_compare_to_full(m,              ComputeThinV, fullSvd) ));
    CALL_SUBTEST(( svd_compare_to_full(m, ComputeThinU|ComputeFullV, fullSvd) ));
    CALL_SUBTEST(( svd_compare_to_full(m, ComputeThinU             , fullSvd) ));
    CALL_SUBTEST(( svd_compare_to_full(m, ComputeThinU|ComputeThinV, fullSvd) ));
    
    CALL_SUBTEST(( svd_least_square<SvdType>(m, ComputeFullU | ComputeThinV) ));
    CALL_SUBTEST(( svd_least_square<SvdType>(m, ComputeThinU | ComputeFullV) ));
    CALL_SUBTEST(( svd_least_square<SvdType>(m, ComputeThinU | ComputeThinV) ));

    CALL_SUBTEST(( svd_min_norm(m, ComputeFullU | ComputeThinV) ));
    CALL_SUBTEST(( svd_min_norm(m, ComputeThinU | ComputeFullV) ));
    CALL_SUBTEST(( svd_min_norm(m, ComputeThinU | ComputeThinV) ));

    // test reconstruction
    Index diagSize = (std::min)(m.rows(), m.cols());
    SvdType svd(m, ComputeThinU | ComputeThinV);
    VERIFY_IS_APPROX(m, svd.matrixU().leftCols(diagSize) * svd.singularValues().asDiagonal() * svd.matrixV().leftCols(diagSize).adjoint());
  }
}


// work around stupid msvc error when constructing at compile time an expression that involves
// a division by zero, even if the numeric type has floating point
template<typename Scalar>
EIGEN_DONT_INLINE Scalar zero() { return Scalar(0); }

// workaround aggressive optimization in ICC
template<typename T> EIGEN_DONT_INLINE  T sub(T a, T b) { return a - b; }

// all this function does is verify we don't iterate infinitely on nan/inf values
template<typename SvdType, typename MatrixType>
void svd_inf_nan()
{
  SvdType svd;
  typedef typename MatrixType::Scalar Scalar;
  Scalar some_inf = Scalar(1) / zero<Scalar>();
  VERIFY(sub(some_inf, some_inf) != sub(some_inf, some_inf));
  svd.compute(MatrixType::Constant(10,10,some_inf), ComputeFullU | ComputeFullV);

  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  VERIFY(nan != nan);
  svd.compute(MatrixType::Constant(10,10,nan), ComputeFullU | ComputeFullV);

  MatrixType m = MatrixType::Zero(10,10);
  m(internal::random<int>(0,9), internal::random<int>(0,9)) = some_inf;
  svd.compute(m, ComputeFullU | ComputeFullV);

  m = MatrixType::Zero(10,10);
  m(internal::random<int>(0,9), internal::random<int>(0,9)) = nan;
  svd.compute(m, ComputeFullU | ComputeFullV);
  
  // regression test for bug 791
  m.resize(3,3);
  m << 0,    2*NumTraits<Scalar>::epsilon(),  0.5,
       0,   -0.5,                             0,
       nan,  0,                               0;
  svd.compute(m, ComputeFullU | ComputeFullV);
  
  m.resize(4,4);
  m <<  1, 0, 0, 0,
        0, 3, 1, 2e-308,
        1, 0, 1, nan,
        0, nan, nan, 0;
  svd.compute(m, ComputeFullU | ComputeFullV);
}

// Regression test for bug 286: JacobiSVD loops indefinitely with some
// matrices containing denormal numbers.
template<typename>
void svd_underoverflow()
{
#if defined __INTEL_COMPILER
// shut up warning #239: floating point underflow
#pragma warning push
#pragma warning disable 239
#endif
  Matrix2d M;
  M << -7.90884e-313, -4.94e-324,
                 0, 5.60844e-313;
  SVD_DEFAULT(Matrix2d) svd;
  svd.compute(M,ComputeFullU|ComputeFullV);
  CALL_SUBTEST( svd_check_full(M,svd) );
  
  // Check all 2x2 matrices made with the following coefficients:
  VectorXd value_set(9);
  value_set << 0, 1, -1, 5.60844e-313, -5.60844e-313, 4.94e-324, -4.94e-324, -4.94e-223, 4.94e-223;
  Array4i id(0,0,0,0);
  int k = 0;
  do
  {
    M << value_set(id(0)), value_set(id(1)), value_set(id(2)), value_set(id(3));
    svd.compute(M,ComputeFullU|ComputeFullV);
    CALL_SUBTEST( svd_check_full(M,svd) );

    id(k)++;
    if(id(k)>=value_set.size())
    {
      while(k<3 && id(k)>=value_set.size()) id(++k)++;
      id.head(k).setZero();
      k=0;
    }

  } while((id<int(value_set.size())).all());
  
#if defined __INTEL_COMPILER
#pragma warning pop
#endif
  
  // Check for overflow:
  Matrix3d M3;
  M3 << 4.4331978442502944e+307, -5.8585363752028680e+307,  6.4527017443412964e+307,
        3.7841695601406358e+307,  2.4331702789740617e+306, -3.5235707140272905e+307,
       -8.7190887618028355e+307, -7.3453213709232193e+307, -2.4367363684472105e+307;

  SVD_DEFAULT(Matrix3d) svd3;
  svd3.compute(M3,ComputeFullU|ComputeFullV); // just check we don't loop indefinitely
  CALL_SUBTEST( svd_check_full(M3,svd3) );
}

// void jacobisvd(const MatrixType& a = MatrixType(), bool pickrandom = true)

template<typename MatrixType>
void svd_all_trivial_2x2( void (*cb)(const MatrixType&,bool) )
{
  MatrixType M;
  VectorXd value_set(3);
  value_set << 0, 1, -1;
  Array4i id(0,0,0,0);
  int k = 0;
  do
  {
    M << value_set(id(0)), value_set(id(1)), value_set(id(2)), value_set(id(3));
    
    cb(M,false);
    
    id(k)++;
    if(id(k)>=value_set.size())
    {
      while(k<3 && id(k)>=value_set.size()) id(++k)++;
      id.head(k).setZero();
      k=0;
    }
    
  } while((id<int(value_set.size())).all());
}

template<typename>
void svd_preallocate()
{
  Vector3f v(3.f, 2.f, 1.f);
  MatrixXf m = v.asDiagonal();

  internal::set_is_malloc_allowed(false);
  VERIFY_RAISES_ASSERT(VectorXf tmp(10);)
  SVD_DEFAULT(MatrixXf) svd;
  internal::set_is_malloc_allowed(true);
  svd.compute(m);
  VERIFY_IS_APPROX(svd.singularValues(), v);

  SVD_DEFAULT(MatrixXf) svd2(3,3);
  internal::set_is_malloc_allowed(false);
  svd2.compute(m);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_APPROX(svd2.singularValues(), v);
  VERIFY_RAISES_ASSERT(svd2.matrixU());
  VERIFY_RAISES_ASSERT(svd2.matrixV());
  svd2.compute(m, ComputeFullU | ComputeFullV);
  VERIFY_IS_APPROX(svd2.matrixU(), Matrix3f::Identity());
  VERIFY_IS_APPROX(svd2.matrixV(), Matrix3f::Identity());
  internal::set_is_malloc_allowed(false);
  svd2.compute(m);
  internal::set_is_malloc_allowed(true);

  SVD_DEFAULT(MatrixXf) svd3(3,3,ComputeFullU|ComputeFullV);
  internal::set_is_malloc_allowed(false);
  svd2.compute(m);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_APPROX(svd2.singularValues(), v);
  VERIFY_IS_APPROX(svd2.matrixU(), Matrix3f::Identity());
  VERIFY_IS_APPROX(svd2.matrixV(), Matrix3f::Identity());
  internal::set_is_malloc_allowed(false);
  svd2.compute(m, ComputeFullU|ComputeFullV);
  internal::set_is_malloc_allowed(true);
}

template<typename SvdType,typename MatrixType> 
void svd_verify_assert(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  Index rows = m.rows();
  Index cols = m.cols();

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };

  typedef Matrix<Scalar, RowsAtCompileTime, 1> RhsType;
  RhsType rhs(rows);
  SvdType svd;
  VERIFY_RAISES_ASSERT(svd.matrixU())
  VERIFY_RAISES_ASSERT(svd.singularValues())
  VERIFY_RAISES_ASSERT(svd.matrixV())
  VERIFY_RAISES_ASSERT(svd.solve(rhs))
  MatrixType a = MatrixType::Zero(rows, cols);
  a.setZero();
  svd.compute(a, 0);
  VERIFY_RAISES_ASSERT(svd.matrixU())
  VERIFY_RAISES_ASSERT(svd.matrixV())
  svd.singularValues();
  VERIFY_RAISES_ASSERT(svd.solve(rhs))
    
  if (ColsAtCompileTime == Dynamic)
  {
    svd.compute(a, ComputeThinU);
    svd.matrixU();
    VERIFY_RAISES_ASSERT(svd.matrixV())
    VERIFY_RAISES_ASSERT(svd.solve(rhs))
    svd.compute(a, ComputeThinV);
    svd.matrixV();
    VERIFY_RAISES_ASSERT(svd.matrixU())
    VERIFY_RAISES_ASSERT(svd.solve(rhs))
  }
  else
  {
    VERIFY_RAISES_ASSERT(svd.compute(a, ComputeThinU))
    VERIFY_RAISES_ASSERT(svd.compute(a, ComputeThinV))
  }
}

#undef SVD_DEFAULT
#undef SVD_FOR_MIN_NORM
