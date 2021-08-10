
#ifndef EIGEN_BENCH_UTIL_H
#define EIGEN_BENCH_UTIL_H

#include <Eigen/Core>
#include "BenchTimer.h"

using namespace std;
using namespace Eigen;

#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/seq.hpp>
#include <boost/preprocessor/array.hpp>
#include <boost/preprocessor/arithmetic.hpp>
#include <boost/preprocessor/comparison.hpp>
#include <boost/preprocessor/punctuation.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/stringize.hpp>

template<typename MatrixType> void initMatrix_random(MatrixType& mat) __attribute__((noinline));
template<typename MatrixType> void initMatrix_random(MatrixType& mat)
{
  mat.setRandom();// = MatrixType::random(mat.rows(), mat.cols());
}

template<typename MatrixType> void initMatrix_identity(MatrixType& mat) __attribute__((noinline));
template<typename MatrixType> void initMatrix_identity(MatrixType& mat)
{
  mat.setIdentity();
}

#ifndef __INTEL_COMPILER
#define DISABLE_SSE_EXCEPTIONS()  { \
  int aux; \
  asm( \
  "stmxcsr   %[aux]           \n\t" \
  "orl       $32832, %[aux]   \n\t" \
  "ldmxcsr   %[aux]           \n\t" \
  : : [aux] "m" (aux)); \
}
#else
#define DISABLE_SSE_EXCEPTIONS()  
#endif

#ifdef BENCH_GMM
#include <gmm/gmm.h>
template <typename EigenMatrixType, typename GmmMatrixType>
void eiToGmm(const EigenMatrixType& src, GmmMatrixType& dst)
{
  dst.resize(src.rows(),src.cols());
  for (int j=0; j<src.cols(); ++j)
    for (int i=0; i<src.rows(); ++i)
      dst(i,j) = src.coeff(i,j);
}
#endif


#ifdef BENCH_GSL
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
template <typename EigenMatrixType>
void eiToGsl(const EigenMatrixType& src, gsl_matrix** dst)
{
  for (int j=0; j<src.cols(); ++j)
    for (int i=0; i<src.rows(); ++i)
      gsl_matrix_set(*dst, i, j, src.coeff(i,j));
}
#endif

#ifdef BENCH_UBLAS
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
template <typename EigenMatrixType, typename UblasMatrixType>
void eiToUblas(const EigenMatrixType& src, UblasMatrixType& dst)
{
  dst.resize(src.rows(),src.cols());
  for (int j=0; j<src.cols(); ++j)
    for (int i=0; i<src.rows(); ++i)
      dst(i,j) = src.coeff(i,j);
}
template <typename EigenType, typename UblasType>
void eiToUblasVec(const EigenType& src, UblasType& dst)
{
  dst.resize(src.size());
  for (int j=0; j<src.size(); ++j)
      dst[j] = src.coeff(j);
}
#endif

#endif // EIGEN_BENCH_UTIL_H
