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

template<typename MatA, typename MatB, typename MatC>
EIGEN_DONT_INLINE
void lazy_gemm(const MatA &A, const MatB &B, MatC &C)
{
//   escape((void*)A.data());
//   escape((void*)B.data());
  C.noalias() += A.lazyProduct(B);
//   escape((void*)C.data());
}

template<int m, int n, int k, int TA>
EIGEN_DONT_INLINE
double bench()
{
  typedef Matrix<Scalar,m,k,TA> MatA;
  typedef Matrix<Scalar,k,n> MatB;
  typedef Matrix<Scalar,m,n> MatC;

  MatA A(m,k);
  MatB B(k,n);
  MatC C(m,n);
  A.setRandom();
  B.setRandom();
  C.setZero();

  BenchTimer t;

  double up = 1e7*4/sizeof(Scalar);
  double tm0 = 10, tm1 = 20;

  double flops = 2. * m * n * k;
  long rep = std::max(10., std::min(10000., up/flops) );
  long tries = std::max(tm0, std::min(tm1, up/flops) );

  BENCH(t, tries, rep, lazy_gemm(A,B,C));

  return 1e-9 * rep * flops / t.best();
}

template<int m, int n, int k>
double bench_t(int t)
{
  if(t)
    return bench<m,n,k,RowMajor>();
  else
    return bench<m,n,k,0>();
}

EIGEN_DONT_INLINE
double bench_mnk(int m, int n, int k, int t)
{
  int id = m*10000 + n*100 + k;
  switch(id) {
    case  10101 : return bench_t< 1, 1, 1>(t); break;
    case  20202 : return bench_t< 2, 2, 2>(t); break;
    case  30303 : return bench_t< 3, 3, 3>(t); break;
    case  40404 : return bench_t< 4, 4, 4>(t); break;
    case  50505 : return bench_t< 5, 5, 5>(t); break;
    case  60606 : return bench_t< 6, 6, 6>(t); break;
    case  70707 : return bench_t< 7, 7, 7>(t); break;
    case  80808 : return bench_t< 8, 8, 8>(t); break;
    case  90909 : return bench_t< 9, 9, 9>(t); break;
    case 101010 : return bench_t<10,10,10>(t); break;
    case 111111 : return bench_t<11,11,11>(t); break;
    case 121212 : return bench_t<12,12,12>(t); break;
  }
  return 0;
}

int main(int argc, char **argv)
{
  std::vector<double> results;
  
  std::ifstream settings("lazy_gemm_settings.txt");
  long m, n, k, t;
  while(settings >> m >> n >> k >> t)
  {
    //std::cerr << "  Testing " << m << " " << n << " " << k << std::endl;
    results.push_back( bench_mnk(m, n, k, t) );
  }
  
  std::cout << RowVectorXd::Map(results.data(), results.size());
  
  return 0;
}
