
#include <Eigen/Sparse>
#include <bench/BenchTimer.h>
#include <set>

using namespace std;
using namespace Eigen;
using namespace Eigen;

#ifndef SIZE
#define SIZE 1024
#endif

#ifndef DENSITY
#define DENSITY 0.01
#endif

#ifndef SCALAR
#define SCALAR double
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
typedef Matrix<Scalar,Dynamic,1> DenseVector;
typedef SparseMatrix<Scalar> EigenSparseMatrix;

void fillMatrix(float density, int rows, int cols,  EigenSparseMatrix& dst)
{
  dst.reserve(double(rows)*cols*density);
  for(int j = 0; j < cols; j++)
  {
    for(int i = 0; i < rows; i++)
    {
      Scalar v = (internal::random<float>(0,1) < density) ? internal::random<Scalar>() : 0;
      if (v!=0)
        dst.insert(i,j) = v;
    }
  }
  dst.finalize();
}

void fillMatrix2(int nnzPerCol, int rows, int cols,  EigenSparseMatrix& dst)
{
//   std::cout << "alloc " << nnzPerCol*cols << "\n";
  dst.reserve(nnzPerCol*cols);
  for(int j = 0; j < cols; j++)
  {
    std::set<int> aux;
    for(int i = 0; i < nnzPerCol; i++)
    {
      int k = internal::random<int>(0,rows-1);
      while (aux.find(k)!=aux.end())
        k = internal::random<int>(0,rows-1);
      aux.insert(k);

      dst.insert(k,j) = internal::random<Scalar>();
    }
  }
  dst.finalize();
}

void eiToDense(const EigenSparseMatrix& src, DenseMatrix& dst)
{
  dst.setZero();
  for (int j=0; j<src.cols(); ++j)
    for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
      dst(it.index(),j) = it.value();
}

#ifndef NOGMM
#include "gmm/gmm.h"
typedef gmm::csc_matrix<Scalar> GmmSparse;
typedef gmm::col_matrix< gmm::wsvector<Scalar> > GmmDynSparse;
void eiToGmm(const EigenSparseMatrix& src, GmmSparse& dst)
{
  GmmDynSparse tmp(src.rows(), src.cols());
  for (int j=0; j<src.cols(); ++j)
    for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
      tmp(it.index(),j) = it.value();
  gmm::copy(tmp, dst);
}
#endif

#ifndef NOMTL
#include <boost/numeric/mtl/mtl.hpp>
typedef mtl::compressed2D<Scalar, mtl::matrix::parameters<mtl::tag::col_major> > MtlSparse;
typedef mtl::compressed2D<Scalar, mtl::matrix::parameters<mtl::tag::row_major> > MtlSparseRowMajor;
void eiToMtl(const EigenSparseMatrix& src, MtlSparse& dst)
{
  mtl::matrix::inserter<MtlSparse> ins(dst);
  for (int j=0; j<src.cols(); ++j)
    for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
      ins[it.index()][j] = it.value();
}
#endif

#ifdef CSPARSE
extern "C" {
#include "cs.h"
}
void eiToCSparse(const EigenSparseMatrix& src, cs* &dst)
{
  cs* aux = cs_spalloc (0, 0, 1, 1, 1);
  for (int j=0; j<src.cols(); ++j)
    for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
      if (!cs_entry(aux, it.index(), j, it.value()))
      {
        std::cout << "cs_entry error\n";
        exit(2);
      }
   dst = cs_compress(aux);
//    cs_spfree(aux);
}
#endif // CSPARSE

#ifndef NOUBLAS
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/operation.hpp>

typedef boost::numeric::ublas::compressed_matrix<Scalar,boost::numeric::ublas::column_major> UBlasSparse;

void eiToUblas(const EigenSparseMatrix& src, UBlasSparse& dst)
{
  dst.resize(src.rows(), src.cols(), false);
  for (int j=0; j<src.cols(); ++j)
    for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
      dst(it.index(),j) = it.value();
}

template <typename EigenType, typename UblasType>
void eiToUblasVec(const EigenType& src, UblasType& dst)
{
  dst.resize(src.size());
  for (int j=0; j<src.size(); ++j)
      dst[j] = src.coeff(j);
}
#endif

#ifdef OSKI
extern "C" {
#include <oski/oski.h>
}
#endif
