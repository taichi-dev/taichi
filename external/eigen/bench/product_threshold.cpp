
#include <iostream>
#include <Eigen/Core>
#include <bench/BenchTimer.h>

using namespace Eigen;
using namespace std;

#define END 9

template<int S> struct map_size { enum { ret = S }; };
template<>  struct map_size<10> { enum { ret = 20 }; };
template<>  struct map_size<11> { enum { ret = 50 }; };
template<>  struct map_size<12> { enum { ret = 100 }; };
template<>  struct map_size<13> { enum { ret = 300 }; };

template<int M, int N,int K> struct alt_prod
{
  enum {
    ret = M==1 && N==1 ? InnerProduct
        : K==1 ? OuterProduct
        : M==1 ? GemvProduct
        : N==1 ? GemvProduct
        : GemmProduct
  };
};
        
void print_mode(int mode)
{
  if(mode==InnerProduct) std::cout << "i";
  if(mode==OuterProduct) std::cout << "o";
  if(mode==CoeffBasedProductMode) std::cout << "c";
  if(mode==LazyCoeffBasedProductMode) std::cout << "l";
  if(mode==GemvProduct) std::cout << "v";
  if(mode==GemmProduct) std::cout << "m";
}

template<int Mode, typename Lhs, typename Rhs, typename Res>
EIGEN_DONT_INLINE void prod(const Lhs& a, const Rhs& b, Res& c)
{
  c.noalias() += typename ProductReturnType<Lhs,Rhs,Mode>::Type(a,b);
}

template<int M, int N, int K, typename Scalar, int Mode>
EIGEN_DONT_INLINE void bench_prod()
{
  typedef Matrix<Scalar,M,K> Lhs; Lhs a; a.setRandom();
  typedef Matrix<Scalar,K,N> Rhs; Rhs b; b.setRandom();
  typedef Matrix<Scalar,M,N> Res; Res c; c.setRandom();

  BenchTimer t;
  double n = 2.*double(M)*double(N)*double(K);
  int rep = 100000./n;
  rep /= 2;
  if(rep<1) rep = 1;
  do {
    rep *= 2;
    t.reset();
    BENCH(t,1,rep,prod<CoeffBasedProductMode>(a,b,c));
  } while(t.best()<0.1);
  
  t.reset();
  BENCH(t,5,rep,prod<Mode>(a,b,c));

  print_mode(Mode);
  std::cout << int(1e-6*n*rep/t.best()) << "\t";
}

template<int N> struct print_n;
template<int M, int N, int K> struct loop_on_m;
template<int M, int N, int K, typename Scalar, int Mode> struct loop_on_n;

template<int M, int N, int K>
struct loop_on_k
{
  static void run()
  {
    std::cout << "K=" << K << "\t";
    print_n<N>::run();
    std::cout << "\n";

    loop_on_m<M,N,K>::run();
    std::cout << "\n\n";

    loop_on_k<M,N,K+1>::run();
  }
};

template<int M, int N>
struct loop_on_k<M,N,END> { static void run(){} };


template<int M, int N, int K>
struct loop_on_m
{
  static void run()
  {
    std::cout << M << "f\t";
    loop_on_n<M,N,K,float,CoeffBasedProductMode>::run();
    std::cout << "\n";
    
    std::cout << M << "f\t";
    loop_on_n<M,N,K,float,-1>::run();
    std::cout << "\n";

    loop_on_m<M+1,N,K>::run();
  }
};

template<int N, int K>
struct loop_on_m<END,N,K> { static void run(){} };

template<int M, int N, int K, typename Scalar, int Mode>
struct loop_on_n
{
  static void run()
  {
    bench_prod<M,N,K,Scalar,Mode==-1? alt_prod<M,N,K>::ret : Mode>();
    
    loop_on_n<M,N+1,K,Scalar,Mode>::run();
  }
};

template<int M, int K, typename Scalar, int Mode>
struct loop_on_n<M,END,K,Scalar,Mode> { static void run(){} };

template<int N> struct print_n
{
  static void run()
  {
    std::cout << map_size<N>::ret << "\t";
    print_n<N+1>::run();
  }
};

template<> struct print_n<END> { static void run(){} };

int main()
{
  loop_on_k<1,1,1>::run();
  
  return 0; 
}
