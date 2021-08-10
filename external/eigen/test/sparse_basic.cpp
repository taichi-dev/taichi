// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Daniel Gomez Ferro <dgomezferro@gmail.com>
// Copyright (C) 2013 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

static long g_realloc_count = 0;
#define EIGEN_SPARSE_COMPRESSED_STORAGE_REALLOCATE_PLUGIN g_realloc_count++;

#include "sparse.h"

template<typename SparseMatrixType> void sparse_basic(const SparseMatrixType& ref)
{
  typedef typename SparseMatrixType::StorageIndex StorageIndex;
  typedef Matrix<StorageIndex,2,1> Vector2;
  
  const Index rows = ref.rows();
  const Index cols = ref.cols();
  //const Index inner = ref.innerSize();
  //const Index outer = ref.outerSize();

  typedef typename SparseMatrixType::Scalar Scalar;
  typedef typename SparseMatrixType::RealScalar RealScalar;
  enum { Flags = SparseMatrixType::Flags };

  double density = (std::max)(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  Scalar eps = 1e-6;

  Scalar s1 = internal::random<Scalar>();
  {
    SparseMatrixType m(rows, cols);
    DenseMatrix refMat = DenseMatrix::Zero(rows, cols);
    DenseVector vec1 = DenseVector::Random(rows);

    std::vector<Vector2> zeroCoords;
    std::vector<Vector2> nonzeroCoords;
    initSparse<Scalar>(density, refMat, m, 0, &zeroCoords, &nonzeroCoords);

    // test coeff and coeffRef
    for (std::size_t i=0; i<zeroCoords.size(); ++i)
    {
      VERIFY_IS_MUCH_SMALLER_THAN( m.coeff(zeroCoords[i].x(),zeroCoords[i].y()), eps );
      if(internal::is_same<SparseMatrixType,SparseMatrix<Scalar,Flags> >::value)
        VERIFY_RAISES_ASSERT( m.coeffRef(zeroCoords[i].x(),zeroCoords[i].y()) = 5 );
    }
    VERIFY_IS_APPROX(m, refMat);

    if(!nonzeroCoords.empty()) {
      m.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);
      refMat.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);
    }

    VERIFY_IS_APPROX(m, refMat);

      // test assertion
      VERIFY_RAISES_ASSERT( m.coeffRef(-1,1) = 0 );
      VERIFY_RAISES_ASSERT( m.coeffRef(0,m.cols()) = 0 );
    }

    // test insert (inner random)
    {
      DenseMatrix m1(rows,cols);
      m1.setZero();
      SparseMatrixType m2(rows,cols);
      bool call_reserve = internal::random<int>()%2;
      Index nnz = internal::random<int>(1,int(rows)/2);
      if(call_reserve)
      {
        if(internal::random<int>()%2)
          m2.reserve(VectorXi::Constant(m2.outerSize(), int(nnz)));
        else
          m2.reserve(m2.outerSize() * nnz);
      }
      g_realloc_count = 0;
      for (Index j=0; j<cols; ++j)
      {
        for (Index k=0; k<nnz; ++k)
        {
          Index i = internal::random<Index>(0,rows-1);
          if (m1.coeff(i,j)==Scalar(0))
            m2.insert(i,j) = m1(i,j) = internal::random<Scalar>();
        }
      }
      
      if(call_reserve && !SparseMatrixType::IsRowMajor)
      {
        VERIFY(g_realloc_count==0);
      }
      
      m2.finalize();
      VERIFY_IS_APPROX(m2,m1);
    }

    // test insert (fully random)
    {
      DenseMatrix m1(rows,cols);
      m1.setZero();
      SparseMatrixType m2(rows,cols);
      if(internal::random<int>()%2)
        m2.reserve(VectorXi::Constant(m2.outerSize(), 2));
      for (int k=0; k<rows*cols; ++k)
      {
        Index i = internal::random<Index>(0,rows-1);
        Index j = internal::random<Index>(0,cols-1);
        if ((m1.coeff(i,j)==Scalar(0)) && (internal::random<int>()%2))
          m2.insert(i,j) = m1(i,j) = internal::random<Scalar>();
        else
        {
          Scalar v = internal::random<Scalar>();
          m2.coeffRef(i,j) += v;
          m1(i,j) += v;
        }
      }
      VERIFY_IS_APPROX(m2,m1);
    }
    
    // test insert (un-compressed)
    for(int mode=0;mode<4;++mode)
    {
      DenseMatrix m1(rows,cols);
      m1.setZero();
      SparseMatrixType m2(rows,cols);
      VectorXi r(VectorXi::Constant(m2.outerSize(), ((mode%2)==0) ? int(m2.innerSize()) : std::max<int>(1,int(m2.innerSize())/8)));
      m2.reserve(r);
      for (Index k=0; k<rows*cols; ++k)
      {
        Index i = internal::random<Index>(0,rows-1);
        Index j = internal::random<Index>(0,cols-1);
        if (m1.coeff(i,j)==Scalar(0))
          m2.insert(i,j) = m1(i,j) = internal::random<Scalar>();
        if(mode==3)
          m2.reserve(r);
      }
      if(internal::random<int>()%2)
        m2.makeCompressed();
      VERIFY_IS_APPROX(m2,m1);
    }

  // test basic computations
  {
    DenseMatrix refM1 = DenseMatrix::Zero(rows, cols);
    DenseMatrix refM2 = DenseMatrix::Zero(rows, cols);
    DenseMatrix refM3 = DenseMatrix::Zero(rows, cols);
    DenseMatrix refM4 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m1(rows, cols);
    SparseMatrixType m2(rows, cols);
    SparseMatrixType m3(rows, cols);
    SparseMatrixType m4(rows, cols);
    initSparse<Scalar>(density, refM1, m1);
    initSparse<Scalar>(density, refM2, m2);
    initSparse<Scalar>(density, refM3, m3);
    initSparse<Scalar>(density, refM4, m4);

    if(internal::random<bool>())
      m1.makeCompressed();

    Index m1_nnz = m1.nonZeros();

    VERIFY_IS_APPROX(m1*s1, refM1*s1);
    VERIFY_IS_APPROX(m1+m2, refM1+refM2);
    VERIFY_IS_APPROX(m1+m2+m3, refM1+refM2+refM3);
    VERIFY_IS_APPROX(m3.cwiseProduct(m1+m2), refM3.cwiseProduct(refM1+refM2));
    VERIFY_IS_APPROX(m1*s1-m2, refM1*s1-refM2);
    VERIFY_IS_APPROX(m4=m1/s1, refM1/s1);
    VERIFY_IS_EQUAL(m4.nonZeros(), m1_nnz);

    if(SparseMatrixType::IsRowMajor)
      VERIFY_IS_APPROX(m1.innerVector(0).dot(refM2.row(0)), refM1.row(0).dot(refM2.row(0)));
    else
      VERIFY_IS_APPROX(m1.innerVector(0).dot(refM2.col(0)), refM1.col(0).dot(refM2.col(0)));

    DenseVector rv = DenseVector::Random(m1.cols());
    DenseVector cv = DenseVector::Random(m1.rows());
    Index r = internal::random<Index>(0,m1.rows()-2);
    Index c = internal::random<Index>(0,m1.cols()-1);
    VERIFY_IS_APPROX(( m1.template block<1,Dynamic>(r,0,1,m1.cols()).dot(rv)) , refM1.row(r).dot(rv));
    VERIFY_IS_APPROX(m1.row(r).dot(rv), refM1.row(r).dot(rv));
    VERIFY_IS_APPROX(m1.col(c).dot(cv), refM1.col(c).dot(cv));

    VERIFY_IS_APPROX(m1.conjugate(), refM1.conjugate());
    VERIFY_IS_APPROX(m1.real(), refM1.real());

    refM4.setRandom();
    // sparse cwise* dense
    VERIFY_IS_APPROX(m3.cwiseProduct(refM4), refM3.cwiseProduct(refM4));
    // dense cwise* sparse
    VERIFY_IS_APPROX(refM4.cwiseProduct(m3), refM4.cwiseProduct(refM3));
//     VERIFY_IS_APPROX(m3.cwise()/refM4, refM3.cwise()/refM4);

    VERIFY_IS_APPROX(refM4 + m3, refM4 + refM3);
    VERIFY_IS_APPROX(m3 + refM4, refM3 + refM4);
    VERIFY_IS_APPROX(refM4 - m3, refM4 - refM3);
    VERIFY_IS_APPROX(m3 - refM4, refM3 - refM4);
    VERIFY_IS_APPROX((RealScalar(0.5)*refM4 + RealScalar(0.5)*m3).eval(), RealScalar(0.5)*refM4 + RealScalar(0.5)*refM3);
    VERIFY_IS_APPROX((RealScalar(0.5)*refM4 + m3*RealScalar(0.5)).eval(), RealScalar(0.5)*refM4 + RealScalar(0.5)*refM3);
    VERIFY_IS_APPROX((RealScalar(0.5)*refM4 + m3.cwiseProduct(m3)).eval(), RealScalar(0.5)*refM4 + refM3.cwiseProduct(refM3));

    VERIFY_IS_APPROX((RealScalar(0.5)*refM4 + RealScalar(0.5)*m3).eval(), RealScalar(0.5)*refM4 + RealScalar(0.5)*refM3);
    VERIFY_IS_APPROX((RealScalar(0.5)*refM4 + m3*RealScalar(0.5)).eval(), RealScalar(0.5)*refM4 + RealScalar(0.5)*refM3);
    VERIFY_IS_APPROX((RealScalar(0.5)*refM4 + (m3+m3)).eval(), RealScalar(0.5)*refM4 + (refM3+refM3));
    VERIFY_IS_APPROX(((refM3+m3)+RealScalar(0.5)*m3).eval(), RealScalar(0.5)*refM3 + (refM3+refM3));
    VERIFY_IS_APPROX((RealScalar(0.5)*refM4 + (refM3+m3)).eval(), RealScalar(0.5)*refM4 + (refM3+refM3));
    VERIFY_IS_APPROX((RealScalar(0.5)*refM4 + (m3+refM3)).eval(), RealScalar(0.5)*refM4 + (refM3+refM3));


    VERIFY_IS_APPROX(m1.sum(), refM1.sum());

    m4 = m1; refM4 = m4;

    VERIFY_IS_APPROX(m1*=s1, refM1*=s1);
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    VERIFY_IS_APPROX(m1/=s1, refM1/=s1);
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);

    VERIFY_IS_APPROX(m1+=m2, refM1+=refM2);
    VERIFY_IS_APPROX(m1-=m2, refM1-=refM2);

    if (rows>=2 && cols>=2)
    {
      VERIFY_RAISES_ASSERT( m1 += m1.innerVector(0) );
      VERIFY_RAISES_ASSERT( m1 -= m1.innerVector(0) );
      VERIFY_RAISES_ASSERT( refM1 -= m1.innerVector(0) );
      VERIFY_RAISES_ASSERT( refM1 += m1.innerVector(0) );
    }
    m1 = m4; refM1 = refM4;

    // test aliasing
    VERIFY_IS_APPROX((m1 = -m1), (refM1 = -refM1));
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    m1 = m4; refM1 = refM4;
    VERIFY_IS_APPROX((m1 = m1.transpose()), (refM1 = refM1.transpose().eval()));
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    m1 = m4; refM1 = refM4;
    VERIFY_IS_APPROX((m1 = -m1.transpose()), (refM1 = -refM1.transpose().eval()));
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    m1 = m4; refM1 = refM4;
    VERIFY_IS_APPROX((m1 += -m1), (refM1 += -refM1));
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    m1 = m4; refM1 = refM4;

    if(m1.isCompressed())
    {
      VERIFY_IS_APPROX(m1.coeffs().sum(), m1.sum());
      m1.coeffs() += s1;
      for(Index j = 0; j<m1.outerSize(); ++j)
        for(typename SparseMatrixType::InnerIterator it(m1,j); it; ++it)
          refM1(it.row(), it.col()) += s1;
      VERIFY_IS_APPROX(m1, refM1);
    }

    // and/or
    {
      typedef SparseMatrix<bool, SparseMatrixType::Options, typename SparseMatrixType::StorageIndex> SpBool;
      SpBool mb1 = m1.real().template cast<bool>();
      SpBool mb2 = m2.real().template cast<bool>();
      VERIFY_IS_EQUAL(mb1.template cast<int>().sum(), refM1.real().template cast<bool>().count());
      VERIFY_IS_EQUAL((mb1 && mb2).template cast<int>().sum(), (refM1.real().template cast<bool>() && refM2.real().template cast<bool>()).count());
      VERIFY_IS_EQUAL((mb1 || mb2).template cast<int>().sum(), (refM1.real().template cast<bool>() || refM2.real().template cast<bool>()).count());
      SpBool mb3 = mb1 && mb2;
      if(mb1.coeffs().all() && mb2.coeffs().all())
      {
        VERIFY_IS_EQUAL(mb3.nonZeros(), (refM1.real().template cast<bool>() && refM2.real().template cast<bool>()).count());
      }
    }
  }

  // test reverse iterators
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    std::vector<Scalar> ref_value(m2.innerSize());
    std::vector<Index> ref_index(m2.innerSize());
    if(internal::random<bool>())
      m2.makeCompressed();
    for(Index j = 0; j<m2.outerSize(); ++j)
    {
      Index count_forward = 0;

      for(typename SparseMatrixType::InnerIterator it(m2,j); it; ++it)
      {
        ref_value[ref_value.size()-1-count_forward] = it.value();
        ref_index[ref_index.size()-1-count_forward] = it.index();
        count_forward++;
      }
      Index count_reverse = 0;
      for(typename SparseMatrixType::ReverseInnerIterator it(m2,j); it; --it)
      {
        VERIFY_IS_APPROX( std::abs(ref_value[ref_value.size()-count_forward+count_reverse])+1, std::abs(it.value())+1);
        VERIFY_IS_EQUAL( ref_index[ref_index.size()-count_forward+count_reverse] , it.index());
        count_reverse++;
      }
      VERIFY_IS_EQUAL(count_forward, count_reverse);
    }
  }

  // test transpose
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    VERIFY_IS_APPROX(m2.transpose().eval(), refMat2.transpose().eval());
    VERIFY_IS_APPROX(m2.transpose(), refMat2.transpose());

    VERIFY_IS_APPROX(SparseMatrixType(m2.adjoint()), refMat2.adjoint());
    
    // check isApprox handles opposite storage order
    typename Transpose<SparseMatrixType>::PlainObject m3(m2);
    VERIFY(m2.isApprox(m3));
  }

  // test prune
  {
    SparseMatrixType m2(rows, cols);
    DenseMatrix refM2(rows, cols);
    refM2.setZero();
    int countFalseNonZero = 0;
    int countTrueNonZero = 0;
    m2.reserve(VectorXi::Constant(m2.outerSize(), int(m2.innerSize())));
    for (Index j=0; j<m2.cols(); ++j)
    {
      for (Index i=0; i<m2.rows(); ++i)
      {
        float x = internal::random<float>(0,1);
        if (x<0.1f)
        {
          // do nothing
        }
        else if (x<0.5f)
        {
          countFalseNonZero++;
          m2.insert(i,j) = Scalar(0);
        }
        else
        {
          countTrueNonZero++;
          m2.insert(i,j) = Scalar(1);
          refM2(i,j) = Scalar(1);
        }
      }
    }
    if(internal::random<bool>())
      m2.makeCompressed();
    VERIFY(countFalseNonZero+countTrueNonZero == m2.nonZeros());
    if(countTrueNonZero>0)
      VERIFY_IS_APPROX(m2, refM2);
    m2.prune(Scalar(1));
    VERIFY(countTrueNonZero==m2.nonZeros());
    VERIFY_IS_APPROX(m2, refM2);
  }

  // test setFromTriplets
  {
    typedef Triplet<Scalar,StorageIndex> TripletType;
    std::vector<TripletType> triplets;
    Index ntriplets = rows*cols;
    triplets.reserve(ntriplets);
    DenseMatrix refMat_sum  = DenseMatrix::Zero(rows,cols);
    DenseMatrix refMat_prod = DenseMatrix::Zero(rows,cols);
    DenseMatrix refMat_last = DenseMatrix::Zero(rows,cols);

    for(Index i=0;i<ntriplets;++i)
    {
      StorageIndex r = internal::random<StorageIndex>(0,StorageIndex(rows-1));
      StorageIndex c = internal::random<StorageIndex>(0,StorageIndex(cols-1));
      Scalar v = internal::random<Scalar>();
      triplets.push_back(TripletType(r,c,v));
      refMat_sum(r,c) += v;
      if(std::abs(refMat_prod(r,c))==0)
        refMat_prod(r,c) = v;
      else
        refMat_prod(r,c) *= v;
      refMat_last(r,c) = v;
    }
    SparseMatrixType m(rows,cols);
    m.setFromTriplets(triplets.begin(), triplets.end());
    VERIFY_IS_APPROX(m, refMat_sum);

    m.setFromTriplets(triplets.begin(), triplets.end(), std::multiplies<Scalar>());
    VERIFY_IS_APPROX(m, refMat_prod);
#if (defined(__cplusplus) && __cplusplus >= 201103L)
    m.setFromTriplets(triplets.begin(), triplets.end(), [] (Scalar,Scalar b) { return b; });
    VERIFY_IS_APPROX(m, refMat_last);
#endif
  }
  
  // test Map
  {
    DenseMatrix refMat2(rows, cols), refMat3(rows, cols);
    SparseMatrixType m2(rows, cols), m3(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    initSparse<Scalar>(density, refMat3, m3);
    {
      Map<SparseMatrixType> mapMat2(m2.rows(), m2.cols(), m2.nonZeros(), m2.outerIndexPtr(), m2.innerIndexPtr(), m2.valuePtr(), m2.innerNonZeroPtr());
      Map<SparseMatrixType> mapMat3(m3.rows(), m3.cols(), m3.nonZeros(), m3.outerIndexPtr(), m3.innerIndexPtr(), m3.valuePtr(), m3.innerNonZeroPtr());
      VERIFY_IS_APPROX(mapMat2+mapMat3, refMat2+refMat3);
      VERIFY_IS_APPROX(mapMat2+mapMat3, refMat2+refMat3);
    }
    {
      MappedSparseMatrix<Scalar,SparseMatrixType::Options,StorageIndex> mapMat2(m2.rows(), m2.cols(), m2.nonZeros(), m2.outerIndexPtr(), m2.innerIndexPtr(), m2.valuePtr(), m2.innerNonZeroPtr());
      MappedSparseMatrix<Scalar,SparseMatrixType::Options,StorageIndex> mapMat3(m3.rows(), m3.cols(), m3.nonZeros(), m3.outerIndexPtr(), m3.innerIndexPtr(), m3.valuePtr(), m3.innerNonZeroPtr());
      VERIFY_IS_APPROX(mapMat2+mapMat3, refMat2+refMat3);
      VERIFY_IS_APPROX(mapMat2+mapMat3, refMat2+refMat3);
    }

    Index i = internal::random<Index>(0,rows-1);
    Index j = internal::random<Index>(0,cols-1);
    m2.coeffRef(i,j) = 123;
    if(internal::random<bool>())
      m2.makeCompressed();
    Map<SparseMatrixType> mapMat2(rows, cols, m2.nonZeros(), m2.outerIndexPtr(), m2.innerIndexPtr(), m2.valuePtr(),  m2.innerNonZeroPtr());
    VERIFY_IS_EQUAL(m2.coeff(i,j),Scalar(123));
    VERIFY_IS_EQUAL(mapMat2.coeff(i,j),Scalar(123));
    mapMat2.coeffRef(i,j) = -123;
    VERIFY_IS_EQUAL(m2.coeff(i,j),Scalar(-123));
  }

  // test triangularView
  {
    DenseMatrix refMat2(rows, cols), refMat3(rows, cols);
    SparseMatrixType m2(rows, cols), m3(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    refMat3 = refMat2.template triangularView<Lower>();
    m3 = m2.template triangularView<Lower>();
    VERIFY_IS_APPROX(m3, refMat3);

    refMat3 = refMat2.template triangularView<Upper>();
    m3 = m2.template triangularView<Upper>();
    VERIFY_IS_APPROX(m3, refMat3);

    {
      refMat3 = refMat2.template triangularView<UnitUpper>();
      m3 = m2.template triangularView<UnitUpper>();
      VERIFY_IS_APPROX(m3, refMat3);

      refMat3 = refMat2.template triangularView<UnitLower>();
      m3 = m2.template triangularView<UnitLower>();
      VERIFY_IS_APPROX(m3, refMat3);
    }

    refMat3 = refMat2.template triangularView<StrictlyUpper>();
    m3 = m2.template triangularView<StrictlyUpper>();
    VERIFY_IS_APPROX(m3, refMat3);

    refMat3 = refMat2.template triangularView<StrictlyLower>();
    m3 = m2.template triangularView<StrictlyLower>();
    VERIFY_IS_APPROX(m3, refMat3);

    // check sparse-triangular to dense
    refMat3 = m2.template triangularView<StrictlyUpper>();
    VERIFY_IS_APPROX(refMat3, DenseMatrix(refMat2.template triangularView<StrictlyUpper>()));
  }
  
  // test selfadjointView
  if(!SparseMatrixType::IsRowMajor)
  {
    DenseMatrix refMat2(rows, rows), refMat3(rows, rows);
    SparseMatrixType m2(rows, rows), m3(rows, rows);
    initSparse<Scalar>(density, refMat2, m2);
    refMat3 = refMat2.template selfadjointView<Lower>();
    m3 = m2.template selfadjointView<Lower>();
    VERIFY_IS_APPROX(m3, refMat3);

    refMat3 += refMat2.template selfadjointView<Lower>();
    m3 += m2.template selfadjointView<Lower>();
    VERIFY_IS_APPROX(m3, refMat3);

    refMat3 -= refMat2.template selfadjointView<Lower>();
    m3 -= m2.template selfadjointView<Lower>();
    VERIFY_IS_APPROX(m3, refMat3);

    // selfadjointView only works for square matrices:
    SparseMatrixType m4(rows, rows+1);
    VERIFY_RAISES_ASSERT(m4.template selfadjointView<Lower>());
    VERIFY_RAISES_ASSERT(m4.template selfadjointView<Upper>());
  }
  
  // test sparseView
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, rows);
    SparseMatrixType m2(rows, rows);
    initSparse<Scalar>(density, refMat2, m2);
    VERIFY_IS_APPROX(m2.eval(), refMat2.sparseView().eval());

    // sparse view on expressions:
    VERIFY_IS_APPROX((s1*m2).eval(), (s1*refMat2).sparseView().eval());
    VERIFY_IS_APPROX((m2+m2).eval(), (refMat2+refMat2).sparseView().eval());
    VERIFY_IS_APPROX((m2*m2).eval(), (refMat2.lazyProduct(refMat2)).sparseView().eval());
    VERIFY_IS_APPROX((m2*m2).eval(), (refMat2*refMat2).sparseView().eval());
  }

  // test diagonal
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    VERIFY_IS_APPROX(m2.diagonal(), refMat2.diagonal().eval());
    DenseVector d = m2.diagonal();
    VERIFY_IS_APPROX(d, refMat2.diagonal().eval());
    d = m2.diagonal().array();
    VERIFY_IS_APPROX(d, refMat2.diagonal().eval());
    VERIFY_IS_APPROX(const_cast<const SparseMatrixType&>(m2).diagonal(), refMat2.diagonal().eval());
    
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag);
    m2.diagonal()      += refMat2.diagonal();
    refMat2.diagonal() += refMat2.diagonal();
    VERIFY_IS_APPROX(m2, refMat2);
  }
  
  // test diagonal to sparse
  {
    DenseVector d = DenseVector::Random(rows);
    DenseMatrix refMat2 = d.asDiagonal();
    SparseMatrixType m2(rows, rows);
    m2 = d.asDiagonal();
    VERIFY_IS_APPROX(m2, refMat2);
    SparseMatrixType m3(d.asDiagonal());
    VERIFY_IS_APPROX(m3, refMat2);
    refMat2 += d.asDiagonal();
    m2 += d.asDiagonal();
    VERIFY_IS_APPROX(m2, refMat2);
  }
  
  // test conservative resize
  {
      std::vector< std::pair<StorageIndex,StorageIndex> > inc;
      if(rows > 3 && cols > 2)
        inc.push_back(std::pair<StorageIndex,StorageIndex>(-3,-2));
      inc.push_back(std::pair<StorageIndex,StorageIndex>(0,0));
      inc.push_back(std::pair<StorageIndex,StorageIndex>(3,2));
      inc.push_back(std::pair<StorageIndex,StorageIndex>(3,0));
      inc.push_back(std::pair<StorageIndex,StorageIndex>(0,3));
      
      for(size_t i = 0; i< inc.size(); i++) {
        StorageIndex incRows = inc[i].first;
        StorageIndex incCols = inc[i].second;
        SparseMatrixType m1(rows, cols);
        DenseMatrix refMat1 = DenseMatrix::Zero(rows, cols);
        initSparse<Scalar>(density, refMat1, m1);
        
        m1.conservativeResize(rows+incRows, cols+incCols);
        refMat1.conservativeResize(rows+incRows, cols+incCols);
        if (incRows > 0) refMat1.bottomRows(incRows).setZero();
        if (incCols > 0) refMat1.rightCols(incCols).setZero();
        
        VERIFY_IS_APPROX(m1, refMat1);
        
        // Insert new values
        if (incRows > 0) 
          m1.insert(m1.rows()-1, 0) = refMat1(refMat1.rows()-1, 0) = 1;
        if (incCols > 0) 
          m1.insert(0, m1.cols()-1) = refMat1(0, refMat1.cols()-1) = 1;
          
        VERIFY_IS_APPROX(m1, refMat1);
          
          
      }
  }

  // test Identity matrix
  {
    DenseMatrix refMat1 = DenseMatrix::Identity(rows, rows);
    SparseMatrixType m1(rows, rows);
    m1.setIdentity();
    VERIFY_IS_APPROX(m1, refMat1);
    for(int k=0; k<rows*rows/4; ++k)
    {
      Index i = internal::random<Index>(0,rows-1);
      Index j = internal::random<Index>(0,rows-1);
      Scalar v = internal::random<Scalar>();
      m1.coeffRef(i,j) = v;
      refMat1.coeffRef(i,j) = v;
      VERIFY_IS_APPROX(m1, refMat1);
      if(internal::random<Index>(0,10)<2)
        m1.makeCompressed();
    }
    m1.setIdentity();
    refMat1.setIdentity();
    VERIFY_IS_APPROX(m1, refMat1);
  }

  // test array/vector of InnerIterator
  {
    typedef typename SparseMatrixType::InnerIterator IteratorType;

    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    IteratorType static_array[2];
    static_array[0] = IteratorType(m2,0);
    static_array[1] = IteratorType(m2,m2.outerSize()-1);
    VERIFY( static_array[0] || m2.innerVector(static_array[0].outer()).nonZeros() == 0 );
    VERIFY( static_array[1] || m2.innerVector(static_array[1].outer()).nonZeros() == 0 );
    if(static_array[0] && static_array[1])
    {
      ++(static_array[1]);
      static_array[1] = IteratorType(m2,0);
      VERIFY( static_array[1] );
      VERIFY( static_array[1].index() == static_array[0].index() );
      VERIFY( static_array[1].outer() == static_array[0].outer() );
      VERIFY( static_array[1].value() == static_array[0].value() );
    }

    std::vector<IteratorType> iters(2);
    iters[0] = IteratorType(m2,0);
    iters[1] = IteratorType(m2,m2.outerSize()-1);
  }

  // test reserve with empty rows/columns
  {
    SparseMatrixType m1(0,cols);
    m1.reserve(ArrayXi::Constant(m1.outerSize(),1));
    SparseMatrixType m2(rows,0);
    m2.reserve(ArrayXi::Constant(m2.outerSize(),1));
  }
}


template<typename SparseMatrixType>
void big_sparse_triplet(Index rows, Index cols, double density) {
  typedef typename SparseMatrixType::StorageIndex StorageIndex;
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef Triplet<Scalar,Index> TripletType;
  std::vector<TripletType> triplets;
  double nelements = density * rows*cols;
  VERIFY(nelements>=0 && nelements <  NumTraits<StorageIndex>::highest());
  Index ntriplets = Index(nelements);
  triplets.reserve(ntriplets);
  Scalar sum = Scalar(0);
  for(Index i=0;i<ntriplets;++i)
  {
    Index r = internal::random<Index>(0,rows-1);
    Index c = internal::random<Index>(0,cols-1);
    // use positive values to prevent numerical cancellation errors in sum
    Scalar v = numext::abs(internal::random<Scalar>());
    triplets.push_back(TripletType(r,c,v));
    sum += v;
  }
  SparseMatrixType m(rows,cols);
  m.setFromTriplets(triplets.begin(), triplets.end());
  VERIFY(m.nonZeros() <= ntriplets);
  VERIFY_IS_APPROX(sum, m.sum());
}


void test_sparse_basic()
{
  for(int i = 0; i < g_repeat; i++) {
    int r = Eigen::internal::random<int>(1,200), c = Eigen::internal::random<int>(1,200);
    if(Eigen::internal::random<int>(0,4) == 0) {
      r = c; // check square matrices in 25% of tries
    }
    EIGEN_UNUSED_VARIABLE(r+c);
    CALL_SUBTEST_1(( sparse_basic(SparseMatrix<double>(1, 1)) ));
    CALL_SUBTEST_1(( sparse_basic(SparseMatrix<double>(8, 8)) ));
    CALL_SUBTEST_2(( sparse_basic(SparseMatrix<std::complex<double>, ColMajor>(r, c)) ));
    CALL_SUBTEST_2(( sparse_basic(SparseMatrix<std::complex<double>, RowMajor>(r, c)) ));
    CALL_SUBTEST_1(( sparse_basic(SparseMatrix<double>(r, c)) ));
    CALL_SUBTEST_5(( sparse_basic(SparseMatrix<double,ColMajor,long int>(r, c)) ));
    CALL_SUBTEST_5(( sparse_basic(SparseMatrix<double,RowMajor,long int>(r, c)) ));
    
    r = Eigen::internal::random<int>(1,100);
    c = Eigen::internal::random<int>(1,100);
    if(Eigen::internal::random<int>(0,4) == 0) {
      r = c; // check square matrices in 25% of tries
    }
    
    CALL_SUBTEST_6(( sparse_basic(SparseMatrix<double,ColMajor,short int>(short(r), short(c))) ));
    CALL_SUBTEST_6(( sparse_basic(SparseMatrix<double,RowMajor,short int>(short(r), short(c))) ));
  }

  // Regression test for bug 900: (manually insert higher values here, if you have enough RAM):
  CALL_SUBTEST_3((big_sparse_triplet<SparseMatrix<float, RowMajor, int> >(10000, 10000, 0.125)));
  CALL_SUBTEST_4((big_sparse_triplet<SparseMatrix<double, ColMajor, long int> >(10000, 10000, 0.125)));

  // Regression test for bug 1105
#ifdef EIGEN_TEST_PART_7
  {
    int n = Eigen::internal::random<int>(200,600);
    SparseMatrix<std::complex<double>,0, long> mat(n, n);
    std::complex<double> val;

    for(int i=0; i<n; ++i)
    {
      mat.coeffRef(i, i%(n/10)) = val;
      VERIFY(mat.data().allocatedSize()<20*n);
    }
  }
#endif
}
