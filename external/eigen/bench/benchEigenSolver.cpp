
// g++ -DNDEBUG -O3 -I.. benchEigenSolver.cpp  -o benchEigenSolver && ./benchEigenSolver
// options:
//  -DBENCH_GMM
//  -DBENCH_GSL -lgsl /usr/lib/libcblas.so.3
//  -DEIGEN_DONT_VECTORIZE
//  -msse2
//  -DREPEAT=100
//  -DTRIES=10
//  -DSCALAR=double

#include <iostream>

#include <Eigen/Core>
#include <Eigen/QR>
#include <bench/BenchUtil.h>
using namespace Eigen;

#ifndef REPEAT
#define REPEAT 1000
#endif

#ifndef TRIES
#define TRIES 4
#endif

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;

template <typename MatrixType>
__attribute__ ((noinline)) void benchEigenSolver(const MatrixType& m)
{
  int rows = m.rows();
  int cols = m.cols();

  int stdRepeats = std::max(1,int((REPEAT*1000)/(rows*rows*sqrt(rows))));
  int saRepeats = stdRepeats * 4;

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;

  MatrixType a = MatrixType::Random(rows,cols);
  SquareMatrixType covMat =  a * a.adjoint();

  BenchTimer timerSa, timerStd;

  Scalar acc = 0;
  int r = internal::random<int>(0,covMat.rows()-1);
  int c = internal::random<int>(0,covMat.cols()-1);
  {
    SelfAdjointEigenSolver<SquareMatrixType> ei(covMat);
    for (int t=0; t<TRIES; ++t)
    {
      timerSa.start();
      for (int k=0; k<saRepeats; ++k)
      {
        ei.compute(covMat);
        acc += ei.eigenvectors().coeff(r,c);
      }
      timerSa.stop();
    }
  }

  {
    EigenSolver<SquareMatrixType> ei(covMat);
    for (int t=0; t<TRIES; ++t)
    {
      timerStd.start();
      for (int k=0; k<stdRepeats; ++k)
      {
        ei.compute(covMat);
        acc += ei.eigenvectors().coeff(r,c);
      }
      timerStd.stop();
    }
  }

  if (MatrixType::RowsAtCompileTime==Dynamic)
    std::cout << "dyn   ";
  else
    std::cout << "fixed ";
  std::cout << covMat.rows() << " \t"
            << timerSa.value() * REPEAT / saRepeats << "s \t"
            << timerStd.value() * REPEAT / stdRepeats << "s";

  #ifdef BENCH_GMM
  if (MatrixType::RowsAtCompileTime==Dynamic)
  {
    timerSa.reset();
    timerStd.reset();

    gmm::dense_matrix<Scalar> gmmCovMat(covMat.rows(),covMat.cols());
    gmm::dense_matrix<Scalar> eigvect(covMat.rows(),covMat.cols());
    std::vector<Scalar> eigval(covMat.rows());
    eiToGmm(covMat, gmmCovMat);
    for (int t=0; t<TRIES; ++t)
    {
      timerSa.start();
      for (int k=0; k<saRepeats; ++k)
      {
        gmm::symmetric_qr_algorithm(gmmCovMat, eigval, eigvect);
        acc += eigvect(r,c);
      }
      timerSa.stop();
    }
    // the non-selfadjoint solver does not compute the eigen vectors
//     for (int t=0; t<TRIES; ++t)
//     {
//       timerStd.start();
//       for (int k=0; k<stdRepeats; ++k)
//       {
//         gmm::implicit_qr_algorithm(gmmCovMat, eigval, eigvect);
//         acc += eigvect(r,c);
//       }
//       timerStd.stop();
//     }

    std::cout << " | \t"
              << timerSa.value() * REPEAT / saRepeats << "s"
              << /*timerStd.value() * REPEAT / stdRepeats << "s"*/ "   na   ";
  }
  #endif

  #ifdef BENCH_GSL
  if (MatrixType::RowsAtCompileTime==Dynamic)
  {
    timerSa.reset();
    timerStd.reset();

    gsl_matrix* gslCovMat = gsl_matrix_alloc(covMat.rows(),covMat.cols());
    gsl_matrix* gslCopy = gsl_matrix_alloc(covMat.rows(),covMat.cols());
    gsl_matrix* eigvect = gsl_matrix_alloc(covMat.rows(),covMat.cols());
    gsl_vector* eigval  = gsl_vector_alloc(covMat.rows());
    gsl_eigen_symmv_workspace* eisymm = gsl_eigen_symmv_alloc(covMat.rows());
    
    gsl_matrix_complex* eigvectz = gsl_matrix_complex_alloc(covMat.rows(),covMat.cols());
    gsl_vector_complex* eigvalz  = gsl_vector_complex_alloc(covMat.rows());
    gsl_eigen_nonsymmv_workspace* einonsymm = gsl_eigen_nonsymmv_alloc(covMat.rows());
    
    eiToGsl(covMat, &gslCovMat);
    for (int t=0; t<TRIES; ++t)
    {
      timerSa.start();
      for (int k=0; k<saRepeats; ++k)
      {
        gsl_matrix_memcpy(gslCopy,gslCovMat);
        gsl_eigen_symmv(gslCopy, eigval, eigvect, eisymm);
        acc += gsl_matrix_get(eigvect,r,c);
      }
      timerSa.stop();
    }
    for (int t=0; t<TRIES; ++t)
    {
      timerStd.start();
      for (int k=0; k<stdRepeats; ++k)
      {
        gsl_matrix_memcpy(gslCopy,gslCovMat);
        gsl_eigen_nonsymmv(gslCopy, eigvalz, eigvectz, einonsymm);
        acc += GSL_REAL(gsl_matrix_complex_get(eigvectz,r,c));
      }
      timerStd.stop();
    }

    std::cout << " | \t"
              << timerSa.value() * REPEAT / saRepeats << "s \t"
              << timerStd.value() * REPEAT / stdRepeats << "s";

    gsl_matrix_free(gslCovMat);
    gsl_vector_free(gslCopy);
    gsl_matrix_free(eigvect);
    gsl_vector_free(eigval);
    gsl_matrix_complex_free(eigvectz);
    gsl_vector_complex_free(eigvalz);
    gsl_eigen_symmv_free(eisymm);
    gsl_eigen_nonsymmv_free(einonsymm);
  }
  #endif

  std::cout << "\n";
  
  // make sure the compiler does not optimize too much
  if (acc==123)
    std::cout << acc;
}

int main(int argc, char* argv[])
{
  const int dynsizes[] = {4,6,8,12,16,24,32,64,128,256,512,0};
  std::cout << "size            selfadjoint       generic";
  #ifdef BENCH_GMM
  std::cout << "        GMM++          ";
  #endif
  #ifdef BENCH_GSL
  std::cout << "       GSL (double + ATLAS)  ";
  #endif
  std::cout << "\n";
  for (uint i=0; dynsizes[i]>0; ++i)
    benchEigenSolver(Matrix<Scalar,Dynamic,Dynamic>(dynsizes[i],dynsizes[i]));

  benchEigenSolver(Matrix<Scalar,2,2>());
  benchEigenSolver(Matrix<Scalar,3,3>());
  benchEigenSolver(Matrix<Scalar,4,4>());
  benchEigenSolver(Matrix<Scalar,6,6>());
  benchEigenSolver(Matrix<Scalar,8,8>());
  benchEigenSolver(Matrix<Scalar,12,12>());
  benchEigenSolver(Matrix<Scalar,16,16>());
  return 0;
}

