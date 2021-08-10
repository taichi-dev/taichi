// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TESTSPARSE_H
#define EIGEN_TESTSPARSE_H

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include "main.h"

#if EIGEN_GNUC_AT_LEAST(4,0) && !defined __ICC && !defined(__clang__)

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#include <tr1/unordered_map>
#define EIGEN_UNORDERED_MAP_SUPPORT
namespace std {
  using std::tr1::unordered_map;
}
#endif

#ifdef EIGEN_GOOGLEHASH_SUPPORT
  #include <google/sparse_hash_map>
#endif

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/Sparse>

enum {
  ForceNonZeroDiag = 1,
  MakeLowerTriangular = 2,
  MakeUpperTriangular = 4,
  ForceRealDiag = 8
};

/* Initializes both a sparse and dense matrix with same random values,
 * and a ratio of \a density non zero entries.
 * \param flags is a union of ForceNonZeroDiag, MakeLowerTriangular and MakeUpperTriangular
 *        allowing to control the shape of the matrix.
 * \param zeroCoords and nonzeroCoords allows to get the coordinate lists of the non zero,
 *        and zero coefficients respectively.
 */
template<typename Scalar,int Opt1,int Opt2,typename StorageIndex> void
initSparse(double density,
           Matrix<Scalar,Dynamic,Dynamic,Opt1>& refMat,
           SparseMatrix<Scalar,Opt2,StorageIndex>& sparseMat,
           int flags = 0,
           std::vector<Matrix<StorageIndex,2,1> >* zeroCoords = 0,
           std::vector<Matrix<StorageIndex,2,1> >* nonzeroCoords = 0)
{
  enum { IsRowMajor = SparseMatrix<Scalar,Opt2,StorageIndex>::IsRowMajor };
  sparseMat.setZero();
  //sparseMat.reserve(int(refMat.rows()*refMat.cols()*density));
  sparseMat.reserve(VectorXi::Constant(IsRowMajor ? refMat.rows() : refMat.cols(), int((1.5*density)*(IsRowMajor?refMat.cols():refMat.rows()))));
  
  for(Index j=0; j<sparseMat.outerSize(); j++)
  {
    //sparseMat.startVec(j);
    for(Index i=0; i<sparseMat.innerSize(); i++)
    {
      Index ai(i), aj(j);
      if(IsRowMajor)
        std::swap(ai,aj);
      Scalar v = (internal::random<double>(0,1) < density) ? internal::random<Scalar>() : Scalar(0);
      if ((flags&ForceNonZeroDiag) && (i==j))
      {
        // FIXME: the following is too conservative
        v = internal::random<Scalar>()*Scalar(3.);
        v = v*v;
        if(numext::real(v)>0) v += Scalar(5);
        else                  v -= Scalar(5);
      }
      if ((flags & MakeLowerTriangular) && aj>ai)
        v = Scalar(0);
      else if ((flags & MakeUpperTriangular) && aj<ai)
        v = Scalar(0);

      if ((flags&ForceRealDiag) && (i==j))
        v = numext::real(v);

      if (v!=Scalar(0))
      {
        //sparseMat.insertBackByOuterInner(j,i) = v;
        sparseMat.insertByOuterInner(j,i) = v;
        if (nonzeroCoords)
          nonzeroCoords->push_back(Matrix<StorageIndex,2,1> (ai,aj));
      }
      else if (zeroCoords)
      {
        zeroCoords->push_back(Matrix<StorageIndex,2,1> (ai,aj));
      }
      refMat(ai,aj) = v;
    }
  }
  //sparseMat.finalize();
}

template<typename Scalar,int Opt1,int Opt2,typename Index> void
initSparse(double density,
           Matrix<Scalar,Dynamic,Dynamic, Opt1>& refMat,
           DynamicSparseMatrix<Scalar, Opt2, Index>& sparseMat,
           int flags = 0,
           std::vector<Matrix<Index,2,1> >* zeroCoords = 0,
           std::vector<Matrix<Index,2,1> >* nonzeroCoords = 0)
{
  enum { IsRowMajor = DynamicSparseMatrix<Scalar,Opt2,Index>::IsRowMajor };
  sparseMat.setZero();
  sparseMat.reserve(int(refMat.rows()*refMat.cols()*density));
  for(int j=0; j<sparseMat.outerSize(); j++)
  {
    sparseMat.startVec(j); // not needed for DynamicSparseMatrix
    for(int i=0; i<sparseMat.innerSize(); i++)
    {
      int ai(i), aj(j);
      if(IsRowMajor)
        std::swap(ai,aj);
      Scalar v = (internal::random<double>(0,1) < density) ? internal::random<Scalar>() : Scalar(0);
      if ((flags&ForceNonZeroDiag) && (i==j))
      {
        v = internal::random<Scalar>()*Scalar(3.);
        v = v*v + Scalar(5.);
      }
      if ((flags & MakeLowerTriangular) && aj>ai)
        v = Scalar(0);
      else if ((flags & MakeUpperTriangular) && aj<ai)
        v = Scalar(0);

      if ((flags&ForceRealDiag) && (i==j))
        v = numext::real(v);

      if (v!=Scalar(0))
      {
        sparseMat.insertBackByOuterInner(j,i) = v;
        if (nonzeroCoords)
          nonzeroCoords->push_back(Matrix<Index,2,1> (ai,aj));
      }
      else if (zeroCoords)
      {
        zeroCoords->push_back(Matrix<Index,2,1> (ai,aj));
      }
      refMat(ai,aj) = v;
    }
  }
  sparseMat.finalize();
}

template<typename Scalar,int Options,typename Index> void
initSparse(double density,
           Matrix<Scalar,Dynamic,1>& refVec,
           SparseVector<Scalar,Options,Index>& sparseVec,
           std::vector<int>* zeroCoords = 0,
           std::vector<int>* nonzeroCoords = 0)
{
  sparseVec.reserve(int(refVec.size()*density));
  sparseVec.setZero();
  for(int i=0; i<refVec.size(); i++)
  {
    Scalar v = (internal::random<double>(0,1) < density) ? internal::random<Scalar>() : Scalar(0);
    if (v!=Scalar(0))
    {
      sparseVec.insertBack(i) = v;
      if (nonzeroCoords)
        nonzeroCoords->push_back(i);
    }
    else if (zeroCoords)
        zeroCoords->push_back(i);
    refVec[i] = v;
  }
}

template<typename Scalar,int Options,typename Index> void
initSparse(double density,
           Matrix<Scalar,1,Dynamic>& refVec,
           SparseVector<Scalar,Options,Index>& sparseVec,
           std::vector<int>* zeroCoords = 0,
           std::vector<int>* nonzeroCoords = 0)
{
  sparseVec.reserve(int(refVec.size()*density));
  sparseVec.setZero();
  for(int i=0; i<refVec.size(); i++)
  {
    Scalar v = (internal::random<double>(0,1) < density) ? internal::random<Scalar>() : Scalar(0);
    if (v!=Scalar(0))
    {
      sparseVec.insertBack(i) = v;
      if (nonzeroCoords)
        nonzeroCoords->push_back(i);
    }
    else if (zeroCoords)
        zeroCoords->push_back(i);
    refVec[i] = v;
  }
}


#include <unsupported/Eigen/SparseExtra>
#endif // EIGEN_TESTSPARSE_H
