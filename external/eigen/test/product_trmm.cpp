// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename T>
int get_random_size()
{
  const int factor = NumTraits<T>::ReadCost;
  const int max_test_size = EIGEN_TEST_MAX_SIZE>2*factor ? EIGEN_TEST_MAX_SIZE/factor : EIGEN_TEST_MAX_SIZE;
  return internal::random<int>(1,max_test_size);
}

template<typename Scalar, int Mode, int TriOrder, int OtherOrder, int ResOrder, int OtherCols>
void trmm(int rows=get_random_size<Scalar>(),
          int cols=get_random_size<Scalar>(),
          int otherCols = OtherCols==Dynamic?get_random_size<Scalar>():OtherCols)
{
  typedef Matrix<Scalar,Dynamic,Dynamic,TriOrder> TriMatrix;
  typedef Matrix<Scalar,Dynamic,OtherCols,OtherCols==1?ColMajor:OtherOrder> OnTheRight;
  typedef Matrix<Scalar,OtherCols,Dynamic,OtherCols==1?RowMajor:OtherOrder> OnTheLeft;
  
  typedef Matrix<Scalar,Dynamic,OtherCols,OtherCols==1?ColMajor:ResOrder> ResXS;
  typedef Matrix<Scalar,OtherCols,Dynamic,OtherCols==1?RowMajor:ResOrder> ResSX;

  TriMatrix  mat(rows,cols), tri(rows,cols), triTr(cols,rows), s1tri(rows,cols), s1triTr(cols,rows);
  
  OnTheRight  ge_right(cols,otherCols);
  OnTheLeft   ge_left(otherCols,rows);
  ResSX       ge_sx, ge_sx_save;
  ResXS       ge_xs, ge_xs_save;

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>();

  mat.setRandom();
  tri = mat.template triangularView<Mode>();
  triTr = mat.transpose().template triangularView<Mode>();
  s1tri = (s1*mat).template triangularView<Mode>();
  s1triTr = (s1*mat).transpose().template triangularView<Mode>();
  ge_right.setRandom();
  ge_left.setRandom();

  VERIFY_IS_APPROX( ge_xs = mat.template triangularView<Mode>() * ge_right, tri * ge_right);
  VERIFY_IS_APPROX( ge_sx = ge_left * mat.template triangularView<Mode>(), ge_left * tri);
  
  VERIFY_IS_APPROX( ge_xs.noalias() = mat.template triangularView<Mode>() * ge_right, tri * ge_right);
  VERIFY_IS_APPROX( ge_sx.noalias() = ge_left * mat.template triangularView<Mode>(), ge_left * tri);

  if((Mode&UnitDiag)==0)
    VERIFY_IS_APPROX( ge_xs.noalias() = (s1*mat.adjoint()).template triangularView<Mode>() * (s2*ge_left.transpose()), s1*triTr.conjugate() * (s2*ge_left.transpose()));
  
  VERIFY_IS_APPROX( ge_xs.noalias() = (s1*mat.transpose()).template triangularView<Mode>() * (s2*ge_left.transpose()), s1triTr * (s2*ge_left.transpose()));
  VERIFY_IS_APPROX( ge_sx.noalias() = (s2*ge_left) * (s1*mat).template triangularView<Mode>(), (s2*ge_left)*s1tri);

  VERIFY_IS_APPROX( ge_sx.noalias() = ge_right.transpose() * mat.adjoint().template triangularView<Mode>(), ge_right.transpose() * triTr.conjugate());
  VERIFY_IS_APPROX( ge_sx.noalias() = ge_right.adjoint() * mat.adjoint().template triangularView<Mode>(), ge_right.adjoint() * triTr.conjugate());
  
  ge_xs_save = ge_xs;
  if((Mode&UnitDiag)==0)
    VERIFY_IS_APPROX( (ge_xs_save + s1*triTr.conjugate() * (s2*ge_left.adjoint())).eval(), ge_xs.noalias() += (s1*mat.adjoint()).template triangularView<Mode>() * (s2*ge_left.adjoint()) );
  ge_xs_save = ge_xs;
  VERIFY_IS_APPROX( (ge_xs_save + s1triTr * (s2*ge_left.adjoint())).eval(), ge_xs.noalias() += (s1*mat.transpose()).template triangularView<Mode>() * (s2*ge_left.adjoint()) );
  ge_sx.setRandom();
  ge_sx_save = ge_sx;
  if((Mode&UnitDiag)==0)
    VERIFY_IS_APPROX( ge_sx_save - (ge_right.adjoint() * (-s1 * triTr).conjugate()).eval(), ge_sx.noalias() -= (ge_right.adjoint() * (-s1 * mat).adjoint().template triangularView<Mode>()).eval());
  
  if((Mode&UnitDiag)==0)
    VERIFY_IS_APPROX( ge_xs = (s1*mat).adjoint().template triangularView<Mode>() * ge_left.adjoint(), numext::conj(s1) * triTr.conjugate() * ge_left.adjoint());
  VERIFY_IS_APPROX( ge_xs = (s1*mat).transpose().template triangularView<Mode>() * ge_left.adjoint(), s1triTr * ge_left.adjoint());

  // TODO check with sub-matrix expressions ?

  // destination with a non-default inner-stride
  // see bug 1741
  {
    VERIFY_IS_APPROX( ge_xs.noalias() = mat.template triangularView<Mode>() * ge_right, tri * ge_right);
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixX;
    MatrixX buffer(2*ge_xs.rows(),2*ge_xs.cols());
    Map<ResXS,0,Stride<Dynamic,2> > map1(buffer.data(),ge_xs.rows(),ge_xs.cols(),Stride<Dynamic,2>(2*ge_xs.outerStride(),2));
    buffer.setZero();
    VERIFY_IS_APPROX( map1.noalias() = mat.template triangularView<Mode>() * ge_right, tri * ge_right);
  }
}

template<typename Scalar, int Mode, int TriOrder>
void trmv(int rows=get_random_size<Scalar>(), int cols=get_random_size<Scalar>())
{
  trmm<Scalar,Mode,TriOrder,ColMajor,ColMajor,1>(rows,cols,1);
}

template<typename Scalar, int Mode, int TriOrder, int OtherOrder, int ResOrder>
void trmm(int rows=get_random_size<Scalar>(), int cols=get_random_size<Scalar>(), int otherCols = get_random_size<Scalar>())
{
  trmm<Scalar,Mode,TriOrder,OtherOrder,ResOrder,Dynamic>(rows,cols,otherCols);
}

#define CALL_ALL_ORDERS(NB,SCALAR,MODE)                                             \
  EIGEN_CAT(CALL_SUBTEST_,NB)((trmm<SCALAR, MODE, ColMajor,ColMajor,ColMajor>()));  \
  EIGEN_CAT(CALL_SUBTEST_,NB)((trmm<SCALAR, MODE, ColMajor,ColMajor,RowMajor>()));  \
  EIGEN_CAT(CALL_SUBTEST_,NB)((trmm<SCALAR, MODE, ColMajor,RowMajor,ColMajor>()));  \
  EIGEN_CAT(CALL_SUBTEST_,NB)((trmm<SCALAR, MODE, ColMajor,RowMajor,RowMajor>()));  \
  EIGEN_CAT(CALL_SUBTEST_,NB)((trmm<SCALAR, MODE, RowMajor,ColMajor,ColMajor>()));  \
  EIGEN_CAT(CALL_SUBTEST_,NB)((trmm<SCALAR, MODE, RowMajor,ColMajor,RowMajor>()));  \
  EIGEN_CAT(CALL_SUBTEST_,NB)((trmm<SCALAR, MODE, RowMajor,RowMajor,ColMajor>()));  \
  EIGEN_CAT(CALL_SUBTEST_,NB)((trmm<SCALAR, MODE, RowMajor,RowMajor,RowMajor>()));  \
  \
  EIGEN_CAT(CALL_SUBTEST_1,NB)((trmv<SCALAR, MODE, ColMajor>()));                   \
  EIGEN_CAT(CALL_SUBTEST_1,NB)((trmv<SCALAR, MODE, RowMajor>()));

  
#define CALL_ALL(NB,SCALAR)                 \
  CALL_ALL_ORDERS(EIGEN_CAT(1,NB),SCALAR,Upper)          \
  CALL_ALL_ORDERS(EIGEN_CAT(2,NB),SCALAR,UnitUpper)      \
  CALL_ALL_ORDERS(EIGEN_CAT(3,NB),SCALAR,StrictlyUpper)  \
  CALL_ALL_ORDERS(EIGEN_CAT(1,NB),SCALAR,Lower)          \
  CALL_ALL_ORDERS(EIGEN_CAT(2,NB),SCALAR,UnitLower)      \
  CALL_ALL_ORDERS(EIGEN_CAT(3,NB),SCALAR,StrictlyLower)
  

void test_product_trmm()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    CALL_ALL(1,float);                //  EIGEN_SUFFIXES;11;111;21;121;31;131
    CALL_ALL(2,double);               //  EIGEN_SUFFIXES;12;112;22;122;32;132
    CALL_ALL(3,std::complex<float>);  //  EIGEN_SUFFIXES;13;113;23;123;33;133
    CALL_ALL(4,std::complex<double>); //  EIGEN_SUFFIXES;14;114;24;124;34;134
  }
}
