#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include "../../BenchTimer.h"
using namespace Eigen;

#ifndef SCALAR
#error SCALAR must be defined
#endif

typedef SCALAR Scalar;

typedef Matrix<Scalar,Dynamic,Dynamic> Mat;

EIGEN_DONT_INLINE
void gemm(const Mat &A, const Mat &B, Mat &C)
{
  C.noalias() += A * B;
}

EIGEN_DONT_INLINE
double bench(long m, long n, long k)
{
  Mat A(m,k);
  Mat B(k,n);
  Mat C(m,n);
  A.setRandom();
  B.setRandom();
  C.setZero();
  
  BenchTimer t;
  
  double up = 1e8*4/sizeof(Scalar);
  double tm0 = 4, tm1 = 10;
  if(NumTraits<Scalar>::IsComplex)
  {
    up /= 4;
    tm0 = 2;
    tm1 = 4;
  }
  
  double flops = 2. * m * n * k;
  long rep = std::max(1., std::min(100., up/flops) );
  long tries = std::max(tm0, std::min(tm1, up/flops) );
  
  BENCH(t, tries, rep, gemm(A,B,C));
  
  return 1e-9 * rep * flops / t.best();
}

int main(int argc, char **argv)
{
  std::vector<double> results;
  
  std::ifstream settings("gemm_settings.txt");
  long m, n, k;
  while(settings >> m >> n >> k)
  {
    //std::cerr << "  Testing " << m << " " << n << " " << k << std::endl;
    results.push_back( bench(m, n, k) );
  }
  
  std::cout << RowVectorXd::Map(results.data(), results.size());
  
  return 0;
}
