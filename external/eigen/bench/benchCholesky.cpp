
// g++ -DNDEBUG -O3 -I.. benchLLT.cpp  -o benchLLT && ./benchLLT
// options:
//  -DBENCH_GSL -lgsl /usr/lib/libcblas.so.3
//  -DEIGEN_DONT_VECTORIZE
//  -msse2
//  -DREPEAT=100
//  -DTRIES=10
//  -DSCALAR=double

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <bench/BenchUtil.h>
using namespace Eigen;

#ifndef REPEAT
#define REPEAT 10000
#endif

#ifndef TRIES
#define TRIES 10
#endif

typedef float Scalar;

template <typename MatrixType>
__attribute__ ((noinline)) void benchLLT(const MatrixType& m)
{
  int rows = m.rows();
  int cols = m.cols();

  double cost = 0;
  for (int j=0; j<rows; ++j)
  {
    int r = std::max(rows - j -1,0);
    cost += 2*(r*j+r+j);
  }

  int repeats = (REPEAT*1000)/(rows*rows);

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;

  MatrixType a = MatrixType::Random(rows,cols);
  SquareMatrixType covMat =  a * a.adjoint();

  BenchTimer timerNoSqrt, timerSqrt;

  Scalar acc = 0;
  int r = internal::random<int>(0,covMat.rows()-1);
  int c = internal::random<int>(0,covMat.cols()-1);
  for (int t=0; t<TRIES; ++t)
  {
    timerNoSqrt.start();
    for (int k=0; k<repeats; ++k)
    {
      LDLT<SquareMatrixType> cholnosqrt(covMat);
      acc += cholnosqrt.matrixL().coeff(r,c);
    }
    timerNoSqrt.stop();
  }

  for (int t=0; t<TRIES; ++t)
  {
    timerSqrt.start();
    for (int k=0; k<repeats; ++k)
    {
      LLT<SquareMatrixType> chol(covMat);
      acc += chol.matrixL().coeff(r,c);
    }
    timerSqrt.stop();
  }

  if (MatrixType::RowsAtCompileTime==Dynamic)
    std::cout << "dyn   ";
  else
    std::cout << "fixed ";
  std::cout << covMat.rows() << " \t"
            << (timerNoSqrt.best()) / repeats << "s "
            << "(" << 1e-9 * cost*repeats/timerNoSqrt.best() << " GFLOPS)\t"
            << (timerSqrt.best()) / repeats << "s "
            << "(" << 1e-9 * cost*repeats/timerSqrt.best() << " GFLOPS)\n";


  #ifdef BENCH_GSL
  if (MatrixType::RowsAtCompileTime==Dynamic)
  {
    timerSqrt.reset();

    gsl_matrix* gslCovMat = gsl_matrix_alloc(covMat.rows(),covMat.cols());
    gsl_matrix* gslCopy = gsl_matrix_alloc(covMat.rows(),covMat.cols());

    eiToGsl(covMat, &gslCovMat);
    for (int t=0; t<TRIES; ++t)
    {
      timerSqrt.start();
      for (int k=0; k<repeats; ++k)
      {
        gsl_matrix_memcpy(gslCopy,gslCovMat);
        gsl_linalg_cholesky_decomp(gslCopy);
        acc += gsl_matrix_get(gslCopy,r,c);
      }
      timerSqrt.stop();
    }

    std::cout << " | \t"
              << timerSqrt.value() * REPEAT / repeats << "s";

    gsl_matrix_free(gslCovMat);
  }
  #endif
  std::cout << "\n";
  // make sure the compiler does not optimize too much
  if (acc==123)
    std::cout << acc;
}

int main(int argc, char* argv[])
{
  const int dynsizes[] = {4,6,8,16,24,32,49,64,128,256,512,900,1500,0};
  std::cout << "size            LDLT                            LLT";
//   #ifdef BENCH_GSL
//   std::cout << "       GSL (standard + double + ATLAS)  ";
//   #endif
  std::cout << "\n";
  for (int i=0; dynsizes[i]>0; ++i)
    benchLLT(Matrix<Scalar,Dynamic,Dynamic>(dynsizes[i],dynsizes[i]));

  benchLLT(Matrix<Scalar,2,2>());
  benchLLT(Matrix<Scalar,3,3>());
  benchLLT(Matrix<Scalar,4,4>());
  benchLLT(Matrix<Scalar,5,5>());
  benchLLT(Matrix<Scalar,6,6>());
  benchLLT(Matrix<Scalar,7,7>());
  benchLLT(Matrix<Scalar,8,8>());
  benchLLT(Matrix<Scalar,12,12>());
  benchLLT(Matrix<Scalar,16,16>());
  return 0;
}

