
// g++-4.4 bench_gemm.cpp -I .. -O2 -DNDEBUG -lrt -fopenmp && OMP_NUM_THREADS=2  ./a.out
// icpc bench_gemm.cpp -I .. -O3 -DNDEBUG -lrt -openmp  && OMP_NUM_THREADS=2  ./a.out

// Compilation options:
// 
// -DSCALAR=std::complex<double>
// -DSCALARA=double or -DSCALARB=double
// -DHAVE_BLAS
// -DDECOUPLED
//

#include <iostream>
#include <Eigen/Core>
#include <bench/BenchTimer.h>

using namespace std;
using namespace Eigen;

#ifndef SCALAR
// #define SCALAR std::complex<float>
#define SCALAR float
#endif

#ifndef SCALARA
#define SCALARA SCALAR
#endif

#ifndef SCALARB
#define SCALARB SCALAR
#endif

typedef SCALAR Scalar;
typedef NumTraits<Scalar>::Real RealScalar;
typedef Matrix<SCALARA,Dynamic,Dynamic> A;
typedef Matrix<SCALARB,Dynamic,Dynamic> B;
typedef Matrix<Scalar,Dynamic,Dynamic> C;
typedef Matrix<RealScalar,Dynamic,Dynamic> M;

#ifdef HAVE_BLAS

extern "C" {
  #include <Eigen/src/misc/blas.h>
}

static float fone = 1;
static float fzero = 0;
static double done = 1;
static double szero = 0;
static std::complex<float> cfone = 1;
static std::complex<float> cfzero = 0;
static std::complex<double> cdone = 1;
static std::complex<double> cdzero = 0;
static char notrans = 'N';
static char trans = 'T';  
static char nonunit = 'N';
static char lower = 'L';
static char right = 'R';
static int intone = 1;

void blas_gemm(const MatrixXf& a, const MatrixXf& b, MatrixXf& c)
{
  int M = c.rows(); int N = c.cols(); int K = a.cols();
  int lda = a.rows(); int ldb = b.rows(); int ldc = c.rows();

  sgemm_(&notrans,&notrans,&M,&N,&K,&fone,
         const_cast<float*>(a.data()),&lda,
         const_cast<float*>(b.data()),&ldb,&fone,
         c.data(),&ldc);
}

EIGEN_DONT_INLINE void blas_gemm(const MatrixXd& a, const MatrixXd& b, MatrixXd& c)
{
  int M = c.rows(); int N = c.cols(); int K = a.cols();
  int lda = a.rows(); int ldb = b.rows(); int ldc = c.rows();

  dgemm_(&notrans,&notrans,&M,&N,&K,&done,
         const_cast<double*>(a.data()),&lda,
         const_cast<double*>(b.data()),&ldb,&done,
         c.data(),&ldc);
}

void blas_gemm(const MatrixXcf& a, const MatrixXcf& b, MatrixXcf& c)
{
  int M = c.rows(); int N = c.cols(); int K = a.cols();
  int lda = a.rows(); int ldb = b.rows(); int ldc = c.rows();

  cgemm_(&notrans,&notrans,&M,&N,&K,(float*)&cfone,
         const_cast<float*>((const float*)a.data()),&lda,
         const_cast<float*>((const float*)b.data()),&ldb,(float*)&cfone,
         (float*)c.data(),&ldc);
}

void blas_gemm(const MatrixXcd& a, const MatrixXcd& b, MatrixXcd& c)
{
  int M = c.rows(); int N = c.cols(); int K = a.cols();
  int lda = a.rows(); int ldb = b.rows(); int ldc = c.rows();

  zgemm_(&notrans,&notrans,&M,&N,&K,(double*)&cdone,
         const_cast<double*>((const double*)a.data()),&lda,
         const_cast<double*>((const double*)b.data()),&ldb,(double*)&cdone,
         (double*)c.data(),&ldc);
}



#endif

void matlab_cplx_cplx(const M& ar, const M& ai, const M& br, const M& bi, M& cr, M& ci)
{
  cr.noalias() += ar * br;
  cr.noalias() -= ai * bi;
  ci.noalias() += ar * bi;
  ci.noalias() += ai * br;
  // [cr ci] += [ar ai] * br + [-ai ar] * bi
}

void matlab_real_cplx(const M& a, const M& br, const M& bi, M& cr, M& ci)
{
  cr.noalias() += a * br;
  ci.noalias() += a * bi;
}

void matlab_cplx_real(const M& ar, const M& ai, const M& b, M& cr, M& ci)
{
  cr.noalias() += ar * b;
  ci.noalias() += ai * b;
}

template<typename A, typename B, typename C>
EIGEN_DONT_INLINE void gemm(const A& a, const B& b, C& c)
{
 c.noalias() += a * b;
}

int main(int argc, char ** argv)
{
  std::ptrdiff_t l1 = internal::queryL1CacheSize();
  std::ptrdiff_t l2 = internal::queryTopLevelCacheSize();
  std::cout << "L1 cache size     = " << (l1>0 ? l1/1024 : -1) << " KB\n";
  std::cout << "L2/L3 cache size  = " << (l2>0 ? l2/1024 : -1) << " KB\n";
  typedef internal::gebp_traits<Scalar,Scalar> Traits;
  std::cout << "Register blocking = " << Traits::mr << " x " << Traits::nr << "\n";

  int rep = 1;    // number of repetitions per try
  int tries = 2;  // number of tries, we keep the best

  int s = 2048;
  int m = s;
  int n = s;
  int p = s;
  int cache_size1=-1, cache_size2=l2, cache_size3 = 0;

  bool need_help = false;
  for (int i=1; i<argc;)
  {
    if(argv[i][0]=='-')
    {
      if(argv[i][1]=='s')
      {
        ++i;
        s = atoi(argv[i++]);
        m = n = p = s;
        if(argv[i][0]!='-')
        {
          n = atoi(argv[i++]);
          p = atoi(argv[i++]);
        }
      }
      else if(argv[i][1]=='c')
      {
        ++i;
        cache_size1 = atoi(argv[i++]);
        if(argv[i][0]!='-')
        {
          cache_size2 = atoi(argv[i++]);
          if(argv[i][0]!='-')
            cache_size3 = atoi(argv[i++]);
        }
      }
      else if(argv[i][1]=='t')
      {
        ++i;
        tries = atoi(argv[i++]);
      }
      else if(argv[i][1]=='p')
      {
        ++i;
        rep = atoi(argv[i++]);
      }
    }
    else
    {
      need_help = true;
      break;
    }
  }

  if(need_help)
  {
    std::cout << argv[0] << " -s <matrix sizes> -c <cache sizes> -t <nb tries> -p <nb repeats>\n";
    std::cout << "   <matrix sizes> : size\n";
    std::cout << "   <matrix sizes> : rows columns depth\n";
    return 1;
  }

#if EIGEN_VERSION_AT_LEAST(3,2,90)
  if(cache_size1>0)
    setCpuCacheSizes(cache_size1,cache_size2,cache_size3);
#endif
  
  A a(m,p); a.setRandom();
  B b(p,n); b.setRandom();
  C c(m,n); c.setOnes();
  C rc = c;

  std::cout << "Matrix sizes = " << m << "x" << p << " * " << p << "x" << n << "\n";
  std::ptrdiff_t mc(m), nc(n), kc(p);
  internal::computeProductBlockingSizes<Scalar,Scalar>(kc, mc, nc);
  std::cout << "blocking size (mc x kc) = " << mc << " x " << kc << "\n";

  C r = c;

  // check the parallel product is correct
  #if defined EIGEN_HAS_OPENMP
  Eigen::initParallel();
  int procs = omp_get_max_threads();
  if(procs>1)
  {
    #ifdef HAVE_BLAS
    blas_gemm(a,b,r);
    #else
    omp_set_num_threads(1);
    r.noalias() += a * b;
    omp_set_num_threads(procs);
    #endif
    c.noalias() += a * b;
    if(!r.isApprox(c)) std::cerr << "Warning, your parallel product is crap!\n\n";
  }
  #elif defined HAVE_BLAS
    blas_gemm(a,b,r);
    c.noalias() += a * b;
    if(!r.isApprox(c)) {
      std::cout << (r  - c).norm() << "\n";
      std::cerr << "Warning, your product is crap!\n\n";
    }
  #else
    if(1.*m*n*p<2000.*2000*2000)
    {
      gemm(a,b,c);
      r.noalias() += a.cast<Scalar>() .lazyProduct( b.cast<Scalar>() );
      if(!r.isApprox(c)) {
        std::cout << (r  - c).norm() << "\n";
        std::cerr << "Warning, your product is crap!\n\n";
      }
    }
  #endif

  #ifdef HAVE_BLAS
  BenchTimer tblas;
  c = rc;
  BENCH(tblas, tries, rep, blas_gemm(a,b,c));
  std::cout << "blas  cpu         " << tblas.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/tblas.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << tblas.total(CPU_TIMER)  << "s)\n";
  std::cout << "blas  real        " << tblas.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/tblas.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << tblas.total(REAL_TIMER) << "s)\n";
  #endif

  BenchTimer tmt;
  c = rc;
  BENCH(tmt, tries, rep, gemm(a,b,c));
  std::cout << "eigen cpu         " << tmt.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/tmt.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << tmt.total(CPU_TIMER)  << "s)\n";
  std::cout << "eigen real        " << tmt.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/tmt.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << tmt.total(REAL_TIMER) << "s)\n";

  #ifdef EIGEN_HAS_OPENMP
  if(procs>1)
  {
    BenchTimer tmono;
    omp_set_num_threads(1);
    Eigen::setNbThreads(1);
    c = rc;
    BENCH(tmono, tries, rep, gemm(a,b,c));
    std::cout << "eigen mono cpu    " << tmono.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/tmono.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << tmono.total(CPU_TIMER)  << "s)\n";
    std::cout << "eigen mono real   " << tmono.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/tmono.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << tmono.total(REAL_TIMER) << "s)\n";
    std::cout << "mt speed up x" << tmono.best(CPU_TIMER) / tmt.best(REAL_TIMER)  << " => " << (100.0*tmono.best(CPU_TIMER) / tmt.best(REAL_TIMER))/procs << "%\n";
  }
  #endif
  
  if(1.*m*n*p<30*30*30)
  {
      BenchTimer tmt;
      c = rc;
      BENCH(tmt, tries, rep, c.noalias()+=a.lazyProduct(b));
      std::cout << "lazy cpu         " << tmt.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/tmt.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << tmt.total(CPU_TIMER)  << "s)\n";
      std::cout << "lazy real        " << tmt.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/tmt.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << tmt.total(REAL_TIMER) << "s)\n";
  }
  
  #ifdef DECOUPLED
  if((NumTraits<A::Scalar>::IsComplex) && (NumTraits<B::Scalar>::IsComplex))
  {
    M ar(m,p); ar.setRandom();
    M ai(m,p); ai.setRandom();
    M br(p,n); br.setRandom();
    M bi(p,n); bi.setRandom();
    M cr(m,n); cr.setRandom();
    M ci(m,n); ci.setRandom();
    
    BenchTimer t;
    BENCH(t, tries, rep, matlab_cplx_cplx(ar,ai,br,bi,cr,ci));
    std::cout << "\"matlab\" cpu    " << t.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/t.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << t.total(CPU_TIMER)  << "s)\n";
    std::cout << "\"matlab\" real   " << t.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/t.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << t.total(REAL_TIMER) << "s)\n";
  }
  if((!NumTraits<A::Scalar>::IsComplex) && (NumTraits<B::Scalar>::IsComplex))
  {
    M a(m,p);  a.setRandom();
    M br(p,n); br.setRandom();
    M bi(p,n); bi.setRandom();
    M cr(m,n); cr.setRandom();
    M ci(m,n); ci.setRandom();
    
    BenchTimer t;
    BENCH(t, tries, rep, matlab_real_cplx(a,br,bi,cr,ci));
    std::cout << "\"matlab\" cpu    " << t.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/t.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << t.total(CPU_TIMER)  << "s)\n";
    std::cout << "\"matlab\" real   " << t.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/t.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << t.total(REAL_TIMER) << "s)\n";
  }
  if((NumTraits<A::Scalar>::IsComplex) && (!NumTraits<B::Scalar>::IsComplex))
  {
    M ar(m,p); ar.setRandom();
    M ai(m,p); ai.setRandom();
    M b(p,n);  b.setRandom();
    M cr(m,n); cr.setRandom();
    M ci(m,n); ci.setRandom();
    
    BenchTimer t;
    BENCH(t, tries, rep, matlab_cplx_real(ar,ai,b,cr,ci));
    std::cout << "\"matlab\" cpu    " << t.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/t.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << t.total(CPU_TIMER)  << "s)\n";
    std::cout << "\"matlab\" real   " << t.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/t.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << t.total(REAL_TIMER) << "s)\n";
  }
  #endif

  return 0;
}

