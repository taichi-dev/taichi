// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_STATIC_ASSERT
#include "product.h"
#include <Eigen/LU>

// regression test for bug 447
template<int>
void product1x1()
{
  Matrix<float,1,3> matAstatic;
  Matrix<float,3,1> matBstatic;
  matAstatic.setRandom();
  matBstatic.setRandom();
  VERIFY_IS_APPROX( (matAstatic * matBstatic).coeff(0,0), 
                    matAstatic.cwiseProduct(matBstatic.transpose()).sum() );

  MatrixXf matAdynamic(1,3);
  MatrixXf matBdynamic(3,1);
  matAdynamic.setRandom();
  matBdynamic.setRandom();
  VERIFY_IS_APPROX( (matAdynamic * matBdynamic).coeff(0,0), 
                    matAdynamic.cwiseProduct(matBdynamic.transpose()).sum() );
}

template<typename TC, typename TA, typename TB>
const TC& ref_prod(TC &C, const TA &A, const TB &B)
{
  for(Index i=0;i<C.rows();++i)
    for(Index j=0;j<C.cols();++j)
      for(Index k=0;k<A.cols();++k)
        C.coeffRef(i,j) += A.coeff(i,k) * B.coeff(k,j);
  return C;
}

template<typename T, int Rows, int Cols, int Depth, int OC, int OA, int OB>
typename internal::enable_if<! ( (Rows ==1&&Depth!=1&&OA==ColMajor)
                              || (Depth==1&&Rows !=1&&OA==RowMajor)
                              || (Cols ==1&&Depth!=1&&OB==RowMajor)
                              || (Depth==1&&Cols !=1&&OB==ColMajor)
                              || (Rows ==1&&Cols !=1&&OC==ColMajor)
                              || (Cols ==1&&Rows !=1&&OC==RowMajor)),void>::type
test_lazy_single(int rows, int cols, int depth)
{
  Matrix<T,Rows,Depth,OA> A(rows,depth); A.setRandom();
  Matrix<T,Depth,Cols,OB> B(depth,cols); B.setRandom();
  Matrix<T,Rows,Cols,OC>  C(rows,cols);  C.setRandom();
  Matrix<T,Rows,Cols,OC>  D(C);
  VERIFY_IS_APPROX(C+=A.lazyProduct(B), ref_prod(D,A,B));
}

template<typename T, int Rows, int Cols, int Depth, int OC, int OA, int OB>
typename internal::enable_if<  ( (Rows ==1&&Depth!=1&&OA==ColMajor)
                              || (Depth==1&&Rows !=1&&OA==RowMajor)
                              || (Cols ==1&&Depth!=1&&OB==RowMajor)
                              || (Depth==1&&Cols !=1&&OB==ColMajor)
                              || (Rows ==1&&Cols !=1&&OC==ColMajor)
                              || (Cols ==1&&Rows !=1&&OC==RowMajor)),void>::type
test_lazy_single(int, int, int)
{
}

template<typename T, int Rows, int Cols, int Depth>
void test_lazy_all_layout(int rows=Rows, int cols=Cols, int depth=Depth)
{
  CALL_SUBTEST(( test_lazy_single<T,Rows,Cols,Depth,ColMajor,ColMajor,ColMajor>(rows,cols,depth) ));
  CALL_SUBTEST(( test_lazy_single<T,Rows,Cols,Depth,RowMajor,ColMajor,ColMajor>(rows,cols,depth) ));
  CALL_SUBTEST(( test_lazy_single<T,Rows,Cols,Depth,ColMajor,RowMajor,ColMajor>(rows,cols,depth) ));
  CALL_SUBTEST(( test_lazy_single<T,Rows,Cols,Depth,RowMajor,RowMajor,ColMajor>(rows,cols,depth) ));
  CALL_SUBTEST(( test_lazy_single<T,Rows,Cols,Depth,ColMajor,ColMajor,RowMajor>(rows,cols,depth) ));
  CALL_SUBTEST(( test_lazy_single<T,Rows,Cols,Depth,RowMajor,ColMajor,RowMajor>(rows,cols,depth) ));
  CALL_SUBTEST(( test_lazy_single<T,Rows,Cols,Depth,ColMajor,RowMajor,RowMajor>(rows,cols,depth) ));
  CALL_SUBTEST(( test_lazy_single<T,Rows,Cols,Depth,RowMajor,RowMajor,RowMajor>(rows,cols,depth) ));
}

template<typename T>
void test_lazy_l1()
{
  int rows = internal::random<int>(1,12);
  int cols = internal::random<int>(1,12);
  int depth = internal::random<int>(1,12);

  // Inner
  CALL_SUBTEST(( test_lazy_all_layout<T,1,1,1>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,1,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,1,3>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,1,8>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,1,9>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,1,-1>(1,1,depth) ));

  // Outer
  CALL_SUBTEST(( test_lazy_all_layout<T,2,1,1>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,2,1>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,2,2,1>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,3,3,1>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,4,1>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,8,1>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,-1,1>(4,cols) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,7,-1,1>(7,cols) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,-1,8,1>(rows) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,-1,3,1>(rows) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,-1,-1,1>(rows,cols) ));
}

template<typename T>
void test_lazy_l2()
{
  int rows = internal::random<int>(1,12);
  int cols = internal::random<int>(1,12);
  int depth = internal::random<int>(1,12);

  // mat-vec
  CALL_SUBTEST(( test_lazy_all_layout<T,2,1,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,2,1,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,1,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,1,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,5,1,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,1,5>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,1,6>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,6,1,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,8,1,8>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,-1,1,4>(rows) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,1,-1>(4,1,depth) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,-1,1,-1>(rows,1,depth) ));

  // vec-mat
  CALL_SUBTEST(( test_lazy_all_layout<T,1,2,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,2,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,4,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,4,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,5,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,4,5>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,4,6>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,6,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,8,8>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,-1, 4>(1,cols) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1, 4,-1>(1,4,depth) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,1,-1,-1>(1,cols,depth) ));
}

template<typename T>
void test_lazy_l3()
{
  int rows = internal::random<int>(1,12);
  int cols = internal::random<int>(1,12);
  int depth = internal::random<int>(1,12);
  // mat-mat
  CALL_SUBTEST(( test_lazy_all_layout<T,2,4,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,2,6,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,3,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,8,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,5,6,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,2,5>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,7,6>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,6,8,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,8,3,8>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,-1,6,4>(rows) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,3,-1>(4,3,depth) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,-1,6,-1>(rows,6,depth) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,8,2,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,5,2,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,4,2>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,8,4,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,6,5,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,4,5>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,3,4,6>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,2,6,4>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,7,8,8>() ));
  CALL_SUBTEST(( test_lazy_all_layout<T,8,-1, 4>(8,cols) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,3, 4,-1>(3,4,depth) ));
  CALL_SUBTEST(( test_lazy_all_layout<T,4,-1,-1>(4,cols,depth) ));
}

template<typename T,int N,int M,int K>
void test_linear_but_not_vectorizable()
{
  // Check tricky cases for which the result of the product is a vector and thus must exhibit the LinearBit flag,
  // but is not vectorizable along the linear dimension.
  Index n = N==Dynamic ? internal::random<Index>(1,32) : N;
  Index m = M==Dynamic ? internal::random<Index>(1,32) : M;
  Index k = K==Dynamic ? internal::random<Index>(1,32) : K;

  {
    Matrix<T,N,M+1> A; A.setRandom(n,m+1);
    Matrix<T,M*2,K> B; B.setRandom(m*2,k);
    Matrix<T,1,K> C;
    Matrix<T,1,K> R;

    C.noalias() = A.template topLeftCorner<1,M>() * (B.template topRows<M>()+B.template bottomRows<M>());
    R.noalias() = A.template topLeftCorner<1,M>() * (B.template topRows<M>()+B.template bottomRows<M>()).eval();
    VERIFY_IS_APPROX(C,R);
  }

  {
    Matrix<T,M+1,N,RowMajor> A; A.setRandom(m+1,n);
    Matrix<T,K,M*2,RowMajor> B; B.setRandom(k,m*2);
    Matrix<T,K,1> C;
    Matrix<T,K,1> R;

    C.noalias() = (B.template leftCols<M>()+B.template rightCols<M>())        * A.template topLeftCorner<M,1>();
    R.noalias() = (B.template leftCols<M>()+B.template rightCols<M>()).eval() * A.template topLeftCorner<M,1>();
    VERIFY_IS_APPROX(C,R);
  }
}

template<int Rows>
void bug_1311()
{
  Matrix< double, Rows, 2 > A;  A.setRandom();
  Vector2d b = Vector2d::Random() ;
  Matrix<double,Rows,1> res;
  res.noalias() = 1. * (A * b);
  VERIFY_IS_APPROX(res, A*b);
  res.noalias() = 1.*A * b;
  VERIFY_IS_APPROX(res, A*b);
  res.noalias() = (1.*A).lazyProduct(b);
  VERIFY_IS_APPROX(res, A*b);
  res.noalias() = (1.*A).lazyProduct(1.*b);
  VERIFY_IS_APPROX(res, A*b);
  res.noalias() = (A).lazyProduct(1.*b);
  VERIFY_IS_APPROX(res, A*b);
}

void test_product_small()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( product(Matrix<float, 3, 2>()) );
    CALL_SUBTEST_2( product(Matrix<int, 3, 17>()) );
    CALL_SUBTEST_8( product(Matrix<double, 3, 17>()) );
    CALL_SUBTEST_3( product(Matrix3d()) );
    CALL_SUBTEST_4( product(Matrix4d()) );
    CALL_SUBTEST_5( product(Matrix4f()) );
    CALL_SUBTEST_6( product1x1<0>() );

    CALL_SUBTEST_11( test_lazy_l1<float>() );
    CALL_SUBTEST_12( test_lazy_l2<float>() );
    CALL_SUBTEST_13( test_lazy_l3<float>() );

    CALL_SUBTEST_21( test_lazy_l1<double>() );
    CALL_SUBTEST_22( test_lazy_l2<double>() );
    CALL_SUBTEST_23( test_lazy_l3<double>() );

    CALL_SUBTEST_31( test_lazy_l1<std::complex<float> >() );
    CALL_SUBTEST_32( test_lazy_l2<std::complex<float> >() );
    CALL_SUBTEST_33( test_lazy_l3<std::complex<float> >() );

    CALL_SUBTEST_41( test_lazy_l1<std::complex<double> >() );
    CALL_SUBTEST_42( test_lazy_l2<std::complex<double> >() );
    CALL_SUBTEST_43( test_lazy_l3<std::complex<double> >() );

    CALL_SUBTEST_7(( test_linear_but_not_vectorizable<float,2,1,Dynamic>() ));
    CALL_SUBTEST_7(( test_linear_but_not_vectorizable<float,3,1,Dynamic>() ));
    CALL_SUBTEST_7(( test_linear_but_not_vectorizable<float,2,1,16>() ));

    CALL_SUBTEST_6( bug_1311<3>() );
    CALL_SUBTEST_6( bug_1311<5>() );
  }

#ifdef EIGEN_TEST_PART_6
  {
    // test compilation of (outer_product) * vector
    Vector3f v = Vector3f::Random();
    VERIFY_IS_APPROX( (v * v.transpose()) * v, (v * v.transpose()).eval() * v);
  }
  
  {
    // regression test for pull-request #93
    Eigen::Matrix<double, 1, 1> A;  A.setRandom();
    Eigen::Matrix<double, 18, 1> B; B.setRandom();
    Eigen::Matrix<double, 1, 18> C; C.setRandom();
    VERIFY_IS_APPROX(B * A.inverse(), B * A.inverse()[0]);
    VERIFY_IS_APPROX(A.inverse() * C, A.inverse()[0] * C);
  }

  {
    Eigen::Matrix<double, 10, 10> A, B, C;
    A.setRandom();
    C = A;
    for(int k=0; k<79; ++k)
      C = C * A;
    B.noalias() = (((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A)) * ((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A)))
                * (((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A)) * ((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A)));
    VERIFY_IS_APPROX(B,C);
  }
#endif
}
