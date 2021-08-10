// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010-2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType>
bool equalsIdentity(const MatrixType& A)
{
  typedef typename MatrixType::Scalar Scalar;
  Scalar zero = static_cast<Scalar>(0);

  bool offDiagOK = true;
  for (Index i = 0; i < A.rows(); ++i) {
    for (Index j = i+1; j < A.cols(); ++j) {
      offDiagOK = offDiagOK && (A(i,j) == zero);
    }
  }
  for (Index i = 0; i < A.rows(); ++i) {
    for (Index j = 0; j < (std::min)(i, A.cols()); ++j) {
      offDiagOK = offDiagOK && (A(i,j) == zero);
    }
  }

  bool diagOK = (A.diagonal().array() == 1).all();
  return offDiagOK && diagOK;

}

template<typename VectorType>
void check_extremity_accuracy(const VectorType &v, const typename VectorType::Scalar &low, const typename VectorType::Scalar &high)
{
  typedef typename VectorType::Scalar Scalar;
  typedef typename VectorType::RealScalar RealScalar;

  RealScalar prec = internal::is_same<RealScalar,float>::value ? NumTraits<RealScalar>::dummy_precision()*10 : NumTraits<RealScalar>::dummy_precision()/10;
  Index size = v.size();

  if(size<20)
    return;

  for (int i=0; i<size; ++i)
  {
    if(i<5 || i>size-6)
    {
      Scalar ref = (low*RealScalar(size-i-1))/RealScalar(size-1) + (high*RealScalar(i))/RealScalar(size-1);
      if(std::abs(ref)>1)
      {
        if(!internal::isApprox(v(i), ref, prec))
          std::cout << v(i) << " != " << ref << "  ; relative error: " << std::abs((v(i)-ref)/ref) << "  ; required precision: " << prec << "  ; range: " << low << "," << high << "  ; i: " << i << "\n";
        VERIFY(internal::isApprox(v(i), (low*RealScalar(size-i-1))/RealScalar(size-1) + (high*RealScalar(i))/RealScalar(size-1), prec));
      }
    }
  }
}

template<typename VectorType>
void testVectorType(const VectorType& base)
{
  typedef typename VectorType::Scalar Scalar;
  typedef typename VectorType::RealScalar RealScalar;

  const Index size = base.size();
  
  Scalar high = internal::random<Scalar>(-500,500);
  Scalar low = (size == 1 ? high : internal::random<Scalar>(-500,500));
  if (low>high) std::swap(low,high);

  // check low==high
  if(internal::random<float>(0.f,1.f)<0.05f)
    low = high;
  // check abs(low) >> abs(high)
  else if(size>2 && std::numeric_limits<RealScalar>::max_exponent10>0 && internal::random<float>(0.f,1.f)<0.1f)
    low = -internal::random<Scalar>(1,2) * RealScalar(std::pow(RealScalar(10),std::numeric_limits<RealScalar>::max_exponent10/2));

  const Scalar step = ((size == 1) ? 1 : (high-low)/(size-1));

  // check whether the result yields what we expect it to do
  VectorType m(base);
  m.setLinSpaced(size,low,high);

  if(!NumTraits<Scalar>::IsInteger)
  {
    VectorType n(size);
    for (int i=0; i<size; ++i)
      n(i) = low+i*step;
    VERIFY_IS_APPROX(m,n);

    CALL_SUBTEST( check_extremity_accuracy(m, low, high) );
  }

  if((!NumTraits<Scalar>::IsInteger) || ((high-low)>=size && (Index(high-low)%(size-1))==0) || (Index(high-low+1)<size && (size%Index(high-low+1))==0))
  {
    VectorType n(size);
    if((!NumTraits<Scalar>::IsInteger) || (high-low>=size))
      for (int i=0; i<size; ++i)
        n(i) = size==1 ? low : (low + ((high-low)*Scalar(i))/(size-1));
    else
      for (int i=0; i<size; ++i)
        n(i) = size==1 ? low : low + Scalar((double(high-low+1)*double(i))/double(size));
    VERIFY_IS_APPROX(m,n);

    // random access version
    m = VectorType::LinSpaced(size,low,high);
    VERIFY_IS_APPROX(m,n);
    VERIFY( internal::isApprox(m(m.size()-1),high) );
    VERIFY( size==1 || internal::isApprox(m(0),low) );
    VERIFY_IS_EQUAL(m(m.size()-1) , high);
    if(!NumTraits<Scalar>::IsInteger)
      CALL_SUBTEST( check_extremity_accuracy(m, low, high) );
  }

  VERIFY( m(m.size()-1) <= high );
  VERIFY( (m.array() <= high).all() );
  VERIFY( (m.array() >= low).all() );


  VERIFY( m(m.size()-1) >= low );
  if(size>=1)
  {
    VERIFY( internal::isApprox(m(0),low) );
    VERIFY_IS_EQUAL(m(0) , low);
  }

  // check whether everything works with row and col major vectors
  Matrix<Scalar,Dynamic,1> row_vector(size);
  Matrix<Scalar,1,Dynamic> col_vector(size);
  row_vector.setLinSpaced(size,low,high);
  col_vector.setLinSpaced(size,low,high);
  // when using the extended precision (e.g., FPU) the relative error might exceed 1 bit
  // when computing the squared sum in isApprox, thus the 2x factor.
  VERIFY( row_vector.isApprox(col_vector.transpose(), Scalar(2)*NumTraits<Scalar>::epsilon()));

  Matrix<Scalar,Dynamic,1> size_changer(size+50);
  size_changer.setLinSpaced(size,low,high);
  VERIFY( size_changer.size() == size );

  typedef Matrix<Scalar,1,1> ScalarMatrix;
  ScalarMatrix scalar;
  scalar.setLinSpaced(1,low,high);
  VERIFY_IS_APPROX( scalar, ScalarMatrix::Constant(high) );
  VERIFY_IS_APPROX( ScalarMatrix::LinSpaced(1,low,high), ScalarMatrix::Constant(high) );

  // regression test for bug 526 (linear vectorized transversal)
  if (size > 1 && (!NumTraits<Scalar>::IsInteger)) {
    m.tail(size-1).setLinSpaced(low, high);
    VERIFY_IS_APPROX(m(size-1), high);
  }

  // regression test for bug 1383 (LinSpaced with empty size/range)
  {
    Index n0 = VectorType::SizeAtCompileTime==Dynamic ? 0 : VectorType::SizeAtCompileTime;
    low = internal::random<Scalar>();
    m = VectorType::LinSpaced(n0,low,low-1);
    VERIFY(m.size()==n0);

    if(VectorType::SizeAtCompileTime==Dynamic)
    {
      VERIFY_IS_EQUAL(VectorType::LinSpaced(n0,0,Scalar(n0-1)).sum(),Scalar(0));
      VERIFY_IS_EQUAL(VectorType::LinSpaced(n0,low,low-1).sum(),Scalar(0));
    }

    m.setLinSpaced(n0,0,Scalar(n0-1));
    VERIFY(m.size()==n0);
    m.setLinSpaced(n0,low,low-1);
    VERIFY(m.size()==n0);

    // empty range only:
    VERIFY_IS_APPROX(VectorType::LinSpaced(size,low,low),VectorType::Constant(size,low));
    m.setLinSpaced(size,low,low);
    VERIFY_IS_APPROX(m,VectorType::Constant(size,low));

    if(NumTraits<Scalar>::IsInteger)
    {
      VERIFY_IS_APPROX( VectorType::LinSpaced(size,low,Scalar(low+size-1)), VectorType::LinSpaced(size,Scalar(low+size-1),low).reverse() );

      if(VectorType::SizeAtCompileTime==Dynamic)
      {
        // Check negative multiplicator path:
        for(Index k=1; k<5; ++k)
          VERIFY_IS_APPROX( VectorType::LinSpaced(size,low,Scalar(low+(size-1)*k)), VectorType::LinSpaced(size,Scalar(low+(size-1)*k),low).reverse() );
        // Check negative divisor path:
        for(Index k=1; k<5; ++k)
          VERIFY_IS_APPROX( VectorType::LinSpaced(size*k,low,Scalar(low+size-1)), VectorType::LinSpaced(size*k,Scalar(low+size-1),low).reverse() );
      }
    }
  }
}

template<typename MatrixType>
void testMatrixType(const MatrixType& m)
{
  using std::abs;
  const Index rows = m.rows();
  const Index cols = m.cols();
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  Scalar s1;
  do {
    s1 = internal::random<Scalar>();
  } while(abs(s1)<RealScalar(1e-5) && (!NumTraits<Scalar>::IsInteger));

  MatrixType A;
  A.setIdentity(rows, cols);
  VERIFY(equalsIdentity(A));
  VERIFY(equalsIdentity(MatrixType::Identity(rows, cols)));


  A = MatrixType::Constant(rows,cols,s1);
  Index i = internal::random<Index>(0,rows-1);
  Index j = internal::random<Index>(0,cols-1);
  VERIFY_IS_APPROX( MatrixType::Constant(rows,cols,s1)(i,j), s1 );
  VERIFY_IS_APPROX( MatrixType::Constant(rows,cols,s1).coeff(i,j), s1 );
  VERIFY_IS_APPROX( A(i,j), s1 );
}

void test_nullary()
{
  CALL_SUBTEST_1( testMatrixType(Matrix2d()) );
  CALL_SUBTEST_2( testMatrixType(MatrixXcf(internal::random<int>(1,300),internal::random<int>(1,300))) );
  CALL_SUBTEST_3( testMatrixType(MatrixXf(internal::random<int>(1,300),internal::random<int>(1,300))) );
  
  for(int i = 0; i < g_repeat*10; i++) {
    CALL_SUBTEST_4( testVectorType(VectorXd(internal::random<int>(1,30000))) );
    CALL_SUBTEST_5( testVectorType(Vector4d()) );  // regression test for bug 232
    CALL_SUBTEST_6( testVectorType(Vector3d()) );
    CALL_SUBTEST_7( testVectorType(VectorXf(internal::random<int>(1,30000))) );
    CALL_SUBTEST_8( testVectorType(Vector3f()) );
    CALL_SUBTEST_8( testVectorType(Vector4f()) );
    CALL_SUBTEST_8( testVectorType(Matrix<float,8,1>()) );
    CALL_SUBTEST_8( testVectorType(Matrix<float,1,1>()) );

    CALL_SUBTEST_9( testVectorType(VectorXi(internal::random<int>(1,10))) );
    CALL_SUBTEST_9( testVectorType(VectorXi(internal::random<int>(9,300))) );
    CALL_SUBTEST_9( testVectorType(Matrix<int,1,1>()) );
  }

#ifdef EIGEN_TEST_PART_6
  // Assignment of a RowVectorXd to a MatrixXd (regression test for bug #79).
  VERIFY( (MatrixXd(RowVectorXd::LinSpaced(3, 0, 1)) - RowVector3d(0, 0.5, 1)).norm() < std::numeric_limits<double>::epsilon() );
#endif

#ifdef EIGEN_TEST_PART_9
  // Check possible overflow issue
  {
    int n = 60000;
    ArrayXi a1(n), a2(n);
    a1.setLinSpaced(n, 0, n-1);
    for(int i=0; i<n; ++i)
      a2(i) = i;
    VERIFY_IS_APPROX(a1,a2);
  }
#endif

#ifdef EIGEN_TEST_PART_10
  // check some internal logic
  VERIFY((  internal::has_nullary_operator<internal::scalar_constant_op<double> >::value ));
  VERIFY(( !internal::has_unary_operator<internal::scalar_constant_op<double> >::value ));
  VERIFY(( !internal::has_binary_operator<internal::scalar_constant_op<double> >::value ));
  VERIFY((  internal::functor_has_linear_access<internal::scalar_constant_op<double> >::ret ));

  VERIFY(( !internal::has_nullary_operator<internal::scalar_identity_op<double> >::value ));
  VERIFY(( !internal::has_unary_operator<internal::scalar_identity_op<double> >::value ));
  VERIFY((  internal::has_binary_operator<internal::scalar_identity_op<double> >::value ));
  VERIFY(( !internal::functor_has_linear_access<internal::scalar_identity_op<double> >::ret ));

  VERIFY(( !internal::has_nullary_operator<internal::linspaced_op<float,float> >::value ));
  VERIFY((  internal::has_unary_operator<internal::linspaced_op<float,float> >::value ));
  VERIFY(( !internal::has_binary_operator<internal::linspaced_op<float,float> >::value ));
  VERIFY((  internal::functor_has_linear_access<internal::linspaced_op<float,float> >::ret ));

  // Regression unit test for a weird MSVC bug.
  // Search "nullary_wrapper_workaround_msvc" in CoreEvaluators.h for the details.
  // See also traits<Ref>::match.
  {
    MatrixXf A = MatrixXf::Random(3,3);
    Ref<const MatrixXf> R = 2.0*A;
    VERIFY_IS_APPROX(R, A+A);

    Ref<const MatrixXf> R1 = MatrixXf::Random(3,3)+A;

    VectorXi V = VectorXi::Random(3);
    Ref<const VectorXi> R2 = VectorXi::LinSpaced(3,1,3)+V;
    VERIFY_IS_APPROX(R2, V+Vector3i(1,2,3));

    VERIFY((  internal::has_nullary_operator<internal::scalar_constant_op<float> >::value ));
    VERIFY(( !internal::has_unary_operator<internal::scalar_constant_op<float> >::value ));
    VERIFY(( !internal::has_binary_operator<internal::scalar_constant_op<float> >::value ));
    VERIFY((  internal::functor_has_linear_access<internal::scalar_constant_op<float> >::ret ));

    VERIFY(( !internal::has_nullary_operator<internal::linspaced_op<int,int> >::value ));
    VERIFY((  internal::has_unary_operator<internal::linspaced_op<int,int> >::value ));
    VERIFY(( !internal::has_binary_operator<internal::linspaced_op<int,int> >::value ));
    VERIFY((  internal::functor_has_linear_access<internal::linspaced_op<int,int> >::ret ));
  }
#endif
}
