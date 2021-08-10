#include <typeinfo>
#include <iostream>
#include <Eigen/Core>
#include "BenchTimer.h"
using namespace Eigen;
using namespace std;

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar sqsumNorm(T& v)
{
  return v.norm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar stableNorm(T& v)
{
  return v.stableNorm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar hypotNorm(T& v)
{
  return v.hypotNorm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar blueNorm(T& v)
{
  return v.blueNorm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar lapackNorm(T& v)
{
  typedef typename T::Scalar Scalar;
  int n = v.size();
  Scalar scale = 0;
  Scalar ssq = 1;
  for (int i=0;i<n;++i)
  {
    Scalar ax = std::abs(v.coeff(i));
    if (scale >= ax)
    {
      ssq += numext::abs2(ax/scale);
    }
    else
    {
      ssq = Scalar(1) + ssq * numext::abs2(scale/ax);
      scale = ax;
    }
  }
  return scale * std::sqrt(ssq);
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar twopassNorm(T& v)
{
  typedef typename T::Scalar Scalar;
  Scalar s = v.array().abs().maxCoeff();
  return s*(v/s).norm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar bl2passNorm(T& v)
{
  return v.stableNorm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar divacNorm(T& v)
{
  int n =v.size() / 2;
  for (int i=0;i<n;++i)
    v(i) = v(2*i)*v(2*i) + v(2*i+1)*v(2*i+1);
  n = n/2;
  while (n>0)
  {
    for (int i=0;i<n;++i)
      v(i) = v(2*i) + v(2*i+1);
    n = n/2;
  }
  return std::sqrt(v(0));
}

namespace Eigen {
namespace internal {
#ifdef EIGEN_VECTORIZE
Packet4f plt(const Packet4f& a, Packet4f& b) { return _mm_cmplt_ps(a,b); }
Packet2d plt(const Packet2d& a, Packet2d& b) { return _mm_cmplt_pd(a,b); }

Packet4f pandnot(const Packet4f& a, Packet4f& b) { return _mm_andnot_ps(a,b); }
Packet2d pandnot(const Packet2d& a, Packet2d& b) { return _mm_andnot_pd(a,b); }
#endif
}
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar pblueNorm(const T& v)
{
  #ifndef EIGEN_VECTORIZE
  return v.blueNorm();
  #else
  typedef typename T::Scalar Scalar;

  static int nmax = 0;
  static Scalar b1, b2, s1m, s2m, overfl, rbig, relerr;
  int n;

  if(nmax <= 0)
  {
    int nbig, ibeta, it, iemin, iemax, iexp;
    Scalar abig, eps;

    nbig  = std::numeric_limits<int>::max();            // largest integer
    ibeta = std::numeric_limits<Scalar>::radix; //NumTraits<Scalar>::Base;                    // base for floating-point numbers
    it    = std::numeric_limits<Scalar>::digits; //NumTraits<Scalar>::Mantissa;                // number of base-beta digits in mantissa
    iemin = std::numeric_limits<Scalar>::min_exponent;  // minimum exponent
    iemax = std::numeric_limits<Scalar>::max_exponent;  // maximum exponent
    rbig  = std::numeric_limits<Scalar>::max();         // largest floating-point number

    // Check the basic machine-dependent constants.
    if(iemin > 1 - 2*it || 1+it>iemax || (it==2 && ibeta<5)
      || (it<=4 && ibeta <= 3 ) || it<2)
    {
      eigen_assert(false && "the algorithm cannot be guaranteed on this computer");
    }
    iexp  = -((1-iemin)/2);
    b1    = std::pow(ibeta, iexp);  // lower boundary of midrange
    iexp  = (iemax + 1 - it)/2;
    b2    = std::pow(ibeta,iexp);   // upper boundary of midrange

    iexp  = (2-iemin)/2;
    s1m   = std::pow(ibeta,iexp);   // scaling factor for lower range
    iexp  = - ((iemax+it)/2);
    s2m   = std::pow(ibeta,iexp);   // scaling factor for upper range

    overfl  = rbig*s2m;          // overfow boundary for abig
    eps     = std::pow(ibeta, 1-it);
    relerr  = std::sqrt(eps);      // tolerance for neglecting asml
    abig    = 1.0/eps - 1.0;
    if (Scalar(nbig)>abig)  nmax = abig;  // largest safe n
    else                    nmax = nbig;
  }

  typedef typename internal::packet_traits<Scalar>::type Packet;
  const int ps = internal::packet_traits<Scalar>::size;
  Packet pasml = internal::pset1<Packet>(Scalar(0));
  Packet pamed = internal::pset1<Packet>(Scalar(0));
  Packet pabig = internal::pset1<Packet>(Scalar(0));
  Packet ps2m = internal::pset1<Packet>(s2m);
  Packet ps1m = internal::pset1<Packet>(s1m);
  Packet pb2  = internal::pset1<Packet>(b2);
  Packet pb1  = internal::pset1<Packet>(b1);
  for(int j=0; j<v.size(); j+=ps)
  {
    Packet ax = internal::pabs(v.template packet<Aligned>(j));
    Packet ax_s2m = internal::pmul(ax,ps2m);
    Packet ax_s1m = internal::pmul(ax,ps1m);
    Packet maskBig = internal::plt(pb2,ax);
    Packet maskSml = internal::plt(ax,pb1);

//     Packet maskMed = internal::pand(maskSml,maskBig);
//     Packet scale = internal::pset1(Scalar(0));
//     scale = internal::por(scale, internal::pand(maskBig,ps2m));
//     scale = internal::por(scale, internal::pand(maskSml,ps1m));
//     scale = internal::por(scale, internal::pandnot(internal::pset1(Scalar(1)),maskMed));
//     ax = internal::pmul(ax,scale);
//     ax = internal::pmul(ax,ax);
//     pabig = internal::padd(pabig, internal::pand(maskBig, ax));
//     pasml = internal::padd(pasml, internal::pand(maskSml, ax));
//     pamed = internal::padd(pamed, internal::pandnot(ax,maskMed));


    pabig = internal::padd(pabig, internal::pand(maskBig, internal::pmul(ax_s2m,ax_s2m)));
    pasml = internal::padd(pasml, internal::pand(maskSml, internal::pmul(ax_s1m,ax_s1m)));
    pamed = internal::padd(pamed, internal::pandnot(internal::pmul(ax,ax),internal::pand(maskSml,maskBig)));
  }
  Scalar abig = internal::predux(pabig);
  Scalar asml = internal::predux(pasml);
  Scalar amed = internal::predux(pamed);
  if(abig > Scalar(0))
  {
    abig = std::sqrt(abig);
    if(abig > overfl)
    {
      eigen_assert(false && "overflow");
      return rbig;
    }
    if(amed > Scalar(0))
    {
      abig = abig/s2m;
      amed = std::sqrt(amed);
    }
    else
    {
      return abig/s2m;
    }

  }
  else if(asml > Scalar(0))
  {
    if (amed > Scalar(0))
    {
      abig = std::sqrt(amed);
      amed = std::sqrt(asml) / s1m;
    }
    else
    {
      return std::sqrt(asml)/s1m;
    }
  }
  else
  {
    return std::sqrt(amed);
  }
  asml = std::min(abig, amed);
  abig = std::max(abig, amed);
  if(asml <= abig*relerr)
    return abig;
  else
    return abig * std::sqrt(Scalar(1) + numext::abs2(asml/abig));
  #endif
}

#define BENCH_PERF(NRM) { \
  float af = 0; double ad = 0; std::complex<float> ac = 0; \
  Eigen::BenchTimer tf, td, tcf; tf.reset(); td.reset(); tcf.reset();\
  for (int k=0; k<tries; ++k) { \
    tf.start(); \
    for (int i=0; i<iters; ++i) { af += NRM(vf); } \
    tf.stop(); \
  } \
  for (int k=0; k<tries; ++k) { \
    td.start(); \
    for (int i=0; i<iters; ++i) { ad += NRM(vd); } \
    td.stop(); \
  } \
  /*for (int k=0; k<std::max(1,tries/3); ++k) { \
    tcf.start(); \
    for (int i=0; i<iters; ++i) { ac += NRM(vcf); } \
    tcf.stop(); \
  } */\
  std::cout << #NRM << "\t" << tf.value() << "   " << td.value() <<  "    " << tcf.value() << "\n"; \
}

void check_accuracy(double basef, double based, int s)
{
  double yf = basef * std::abs(internal::random<double>());
  double yd = based * std::abs(internal::random<double>());
  VectorXf vf = VectorXf::Ones(s) * yf;
  VectorXd vd = VectorXd::Ones(s) * yd;

  std::cout << "reference\t" << std::sqrt(double(s))*yf << "\t" << std::sqrt(double(s))*yd << "\n";
  std::cout << "sqsumNorm\t" << sqsumNorm(vf) << "\t" << sqsumNorm(vd) << "\n";
  std::cout << "hypotNorm\t" << hypotNorm(vf) << "\t" << hypotNorm(vd) << "\n";
  std::cout << "blueNorm\t" << blueNorm(vf) << "\t" << blueNorm(vd) << "\n";
  std::cout << "pblueNorm\t" << pblueNorm(vf) << "\t" << pblueNorm(vd) << "\n";
  std::cout << "lapackNorm\t" << lapackNorm(vf) << "\t" << lapackNorm(vd) << "\n";
  std::cout << "twopassNorm\t" << twopassNorm(vf) << "\t" << twopassNorm(vd) << "\n";
  std::cout << "bl2passNorm\t" << bl2passNorm(vf) << "\t" << bl2passNorm(vd) << "\n";
}

void check_accuracy_var(int ef0, int ef1, int ed0, int ed1, int s)
{
  VectorXf vf(s);
  VectorXd vd(s);
  for (int i=0; i<s; ++i)
  {
    vf[i] = std::abs(internal::random<double>()) * std::pow(double(10), internal::random<int>(ef0,ef1));
    vd[i] = std::abs(internal::random<double>()) * std::pow(double(10), internal::random<int>(ed0,ed1));
  }

  //std::cout << "reference\t" << internal::sqrt(double(s))*yf << "\t" << internal::sqrt(double(s))*yd << "\n";
  std::cout << "sqsumNorm\t"  << sqsumNorm(vf)  << "\t" << sqsumNorm(vd)  << "\t" << sqsumNorm(vf.cast<long double>()) << "\t" << sqsumNorm(vd.cast<long double>()) << "\n";
  std::cout << "hypotNorm\t"  << hypotNorm(vf)  << "\t" << hypotNorm(vd)  << "\t" << hypotNorm(vf.cast<long double>()) << "\t" << hypotNorm(vd.cast<long double>()) << "\n";
  std::cout << "blueNorm\t"   << blueNorm(vf)   << "\t" << blueNorm(vd)   << "\t" << blueNorm(vf.cast<long double>()) << "\t" << blueNorm(vd.cast<long double>()) << "\n";
  std::cout << "pblueNorm\t"  << pblueNorm(vf)  << "\t" << pblueNorm(vd)  << "\t" << blueNorm(vf.cast<long double>()) << "\t" << blueNorm(vd.cast<long double>()) << "\n";
  std::cout << "lapackNorm\t" << lapackNorm(vf) << "\t" << lapackNorm(vd) << "\t" << lapackNorm(vf.cast<long double>()) << "\t" << lapackNorm(vd.cast<long double>()) << "\n";
  std::cout << "twopassNorm\t" << twopassNorm(vf) << "\t" << twopassNorm(vd) << "\t" << twopassNorm(vf.cast<long double>()) << "\t" << twopassNorm(vd.cast<long double>()) << "\n";
//   std::cout << "bl2passNorm\t" << bl2passNorm(vf) << "\t" << bl2passNorm(vd) << "\t" << bl2passNorm(vf.cast<long double>()) << "\t" << bl2passNorm(vd.cast<long double>()) << "\n";
}

int main(int argc, char** argv)
{
  int tries = 10;
  int iters = 100000;
  double y = 1.1345743233455785456788e12 * internal::random<double>();
  VectorXf v = VectorXf::Ones(1024) * y;

// return 0;
  int s = 10000;
  double basef_ok = 1.1345743233455785456788e15;
  double based_ok = 1.1345743233455785456788e95;

  double basef_under = 1.1345743233455785456788e-27;
  double based_under = 1.1345743233455785456788e-303;

  double basef_over = 1.1345743233455785456788e+27;
  double based_over = 1.1345743233455785456788e+302;

  std::cout.precision(20);

  std::cerr << "\nNo under/overflow:\n";
  check_accuracy(basef_ok, based_ok, s);

  std::cerr << "\nUnderflow:\n";
  check_accuracy(basef_under, based_under, s);

  std::cerr << "\nOverflow:\n";
  check_accuracy(basef_over, based_over, s);

  std::cerr << "\nVarying (over):\n";
  for (int k=0; k<1; ++k)
  {
    check_accuracy_var(20,27,190,302,s);
    std::cout << "\n";
  }

  std::cerr << "\nVarying (under):\n";
  for (int k=0; k<1; ++k)
  {
    check_accuracy_var(-27,20,-302,-190,s);
    std::cout << "\n";
  }

  y = 1;
  std::cout.precision(4);
  int s1 = 1024*1024*32;
  std::cerr << "Performance (out of cache, " << s1 << "):\n";
  {
    int iters = 1;
    VectorXf vf = VectorXf::Random(s1) * y;
    VectorXd vd = VectorXd::Random(s1) * y;
    VectorXcf vcf = VectorXcf::Random(s1) * y;
    BENCH_PERF(sqsumNorm);
    BENCH_PERF(stableNorm);
    BENCH_PERF(blueNorm);
    BENCH_PERF(pblueNorm);
    BENCH_PERF(lapackNorm);
    BENCH_PERF(hypotNorm);
    BENCH_PERF(twopassNorm);
    BENCH_PERF(bl2passNorm);
  }

  std::cerr << "\nPerformance (in cache, " << 512 << "):\n";
  {
    int iters = 100000;
    VectorXf vf = VectorXf::Random(512) * y;
    VectorXd vd = VectorXd::Random(512) * y;
    VectorXcf vcf = VectorXcf::Random(512) * y;
    BENCH_PERF(sqsumNorm);
    BENCH_PERF(stableNorm);
    BENCH_PERF(blueNorm);
    BENCH_PERF(pblueNorm);
    BENCH_PERF(lapackNorm);
    BENCH_PERF(hypotNorm);
    BENCH_PERF(twopassNorm);
    BENCH_PERF(bl2passNorm);
  }
}
