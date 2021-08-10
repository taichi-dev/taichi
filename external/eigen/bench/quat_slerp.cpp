
#include <iostream>
#include <Eigen/Geometry>
#include <bench/BenchTimer.h>
using namespace Eigen;
using namespace std;



template<typename Q>
EIGEN_DONT_INLINE Q nlerp(const Q& a, const Q& b, typename Q::Scalar t)
{
  return Q((a.coeffs() * (1.0-t) + b.coeffs() * t).normalized());
}

template<typename Q>
EIGEN_DONT_INLINE Q slerp_eigen(const Q& a, const Q& b, typename Q::Scalar t)
{
  return a.slerp(t,b);
}

template<typename Q>
EIGEN_DONT_INLINE Q slerp_legacy(const Q& a, const Q& b, typename Q::Scalar t)
{
  typedef typename Q::Scalar Scalar;
  static const Scalar one = Scalar(1) - dummy_precision<Scalar>();
  Scalar d = a.dot(b);
  Scalar absD = internal::abs(d);
  if (absD>=one)
    return a;

  // theta is the angle between the 2 quaternions
  Scalar theta = std::acos(absD);
  Scalar sinTheta = internal::sin(theta);

  Scalar scale0 = internal::sin( ( Scalar(1) - t ) * theta) / sinTheta;
  Scalar scale1 = internal::sin( ( t * theta) ) / sinTheta;
  if (d<0)
    scale1 = -scale1;

  return Q(scale0 * a.coeffs() + scale1 * b.coeffs());
}

template<typename Q>
EIGEN_DONT_INLINE Q slerp_legacy_nlerp(const Q& a, const Q& b, typename Q::Scalar t)
{
  typedef typename Q::Scalar Scalar;
  static const Scalar one = Scalar(1) - epsilon<Scalar>();
  Scalar d = a.dot(b);
  Scalar absD = internal::abs(d);
  
  Scalar scale0;
  Scalar scale1;
  
  if (absD>=one)
  {
    scale0 = Scalar(1) - t;
    scale1 = t;
  }
  else
  {
    // theta is the angle between the 2 quaternions
    Scalar theta = std::acos(absD);
    Scalar sinTheta = internal::sin(theta);

    scale0 = internal::sin( ( Scalar(1) - t ) * theta) / sinTheta;
    scale1 = internal::sin( ( t * theta) ) / sinTheta;
    if (d<0)
      scale1 = -scale1;
  }

  return Q(scale0 * a.coeffs() + scale1 * b.coeffs());
}

template<typename T>
inline T sin_over_x(T x)
{
  if (T(1) + x*x == T(1))
    return T(1);
  else
    return std::sin(x)/x;
}

template<typename Q>
EIGEN_DONT_INLINE Q slerp_rw(const Q& a, const Q& b, typename Q::Scalar t)
{
  typedef typename Q::Scalar Scalar;
  
  Scalar d = a.dot(b);
  Scalar theta;
  if (d<0.0)
    theta = /*M_PI -*/ Scalar(2)*std::asin( (a.coeffs()+b.coeffs()).norm()/2 );
  else
    theta = Scalar(2)*std::asin( (a.coeffs()-b.coeffs()).norm()/2 );
  
  // theta is the angle between the 2 quaternions
//   Scalar theta = std::acos(absD);
  Scalar sinOverTheta = sin_over_x(theta);

  Scalar scale0 = (Scalar(1)-t)*sin_over_x( ( Scalar(1) - t ) * theta) / sinOverTheta;
  Scalar scale1 = t * sin_over_x( ( t * theta) ) / sinOverTheta;
  if (d<0)
    scale1 = -scale1;

  return Quaternion<Scalar>(scale0 * a.coeffs() + scale1 * b.coeffs());
}

template<typename Q>
EIGEN_DONT_INLINE Q slerp_gael(const Q& a, const Q& b, typename Q::Scalar t)
{
  typedef typename Q::Scalar Scalar;
  
  Scalar d = a.dot(b);
  Scalar theta;
//   theta = Scalar(2) * atan2((a.coeffs()-b.coeffs()).norm(),(a.coeffs()+b.coeffs()).norm());
//   if (d<0.0)
//     theta = M_PI-theta;
  
  if (d<0.0)
    theta = /*M_PI -*/ Scalar(2)*std::asin( (-a.coeffs()-b.coeffs()).norm()/2 );
  else
    theta = Scalar(2)*std::asin( (a.coeffs()-b.coeffs()).norm()/2 );
  
  
  Scalar scale0;
  Scalar scale1;
  if(theta*theta-Scalar(6)==-Scalar(6))
  {
    scale0 = Scalar(1) - t;
    scale1 = t;
  }
  else
  {
    Scalar sinTheta = std::sin(theta);
    scale0 = internal::sin( ( Scalar(1) - t ) * theta) / sinTheta;
    scale1 = internal::sin( ( t * theta) ) / sinTheta;
    if (d<0)
      scale1 = -scale1;
  }

  return Quaternion<Scalar>(scale0 * a.coeffs() + scale1 * b.coeffs());
}

int main()
{
  typedef double RefScalar;
  typedef float TestScalar;
  
  typedef Quaternion<RefScalar>  Qd;
  typedef Quaternion<TestScalar> Qf;
  
  unsigned int g_seed = (unsigned int) time(NULL);
  std::cout << g_seed << "\n";
//   g_seed = 1259932496;
  srand(g_seed);
  
  Matrix<RefScalar,Dynamic,1> maxerr(7);
  maxerr.setZero();
  
  Matrix<RefScalar,Dynamic,1> avgerr(7);
  avgerr.setZero();
  
  cout << "double=>float=>double       nlerp        eigen        legacy(snap)         legacy(nlerp)        rightway         gael's criteria\n";
  
  int rep = 100;
  int iters = 40;
  for (int w=0; w<rep; ++w)
  {
    Qf a, b;
    a.coeffs().setRandom();
    a.normalize();
    b.coeffs().setRandom();
    b.normalize();
    
    Qf c[6];
    
    Qd ar(a.cast<RefScalar>());
    Qd br(b.cast<RefScalar>());
    Qd cr;
    
    
    
    cout.precision(8);
    cout << std::scientific;
    for (int i=0; i<iters; ++i)
    {
      RefScalar t = 0.65;
      cr = slerp_rw(ar,br,t);
      
      Qf refc = cr.cast<TestScalar>();
      c[0] = nlerp(a,b,t);
      c[1] = slerp_eigen(a,b,t);
      c[2] = slerp_legacy(a,b,t);
      c[3] = slerp_legacy_nlerp(a,b,t);
      c[4] = slerp_rw(a,b,t);
      c[5] = slerp_gael(a,b,t);
      
      VectorXd err(7);
      err[0] = (cr.coeffs()-refc.cast<RefScalar>().coeffs()).norm();
//       std::cout << err[0] << "    ";
      for (int k=0; k<6; ++k)
      {
        err[k+1] = (c[k].coeffs()-refc.coeffs()).norm();
//         std::cout << err[k+1] << "    ";
      }
      maxerr = maxerr.cwise().max(err);
      avgerr += err;
//       std::cout << "\n";
      b = cr.cast<TestScalar>();
      br = cr;
    }
//     std::cout << "\n";
  }
  avgerr /= RefScalar(rep*iters);
  cout << "\n\nAccuracy:\n"
       << "  max: " << maxerr.transpose() << "\n";
  cout << "  avg: " << avgerr.transpose() << "\n";
  
  // perf bench
  Quaternionf a,b;
  a.coeffs().setRandom();
  a.normalize();
  b.coeffs().setRandom();
  b.normalize();
  //b = a;
  float s = 0.65;
    
  #define BENCH(FUNC) {\
    BenchTimer t; \
    for(int k=0; k<2; ++k) {\
      t.start(); \
      for(int i=0; i<1000000; ++i) \
        FUNC(a,b,s); \
      t.stop(); \
    } \
    cout << "  " << #FUNC << " => \t " << t.value() << "s\n"; \
  }
  
  cout << "\nSpeed:\n" << std::fixed;
  BENCH(nlerp);
  BENCH(slerp_eigen);
  BENCH(slerp_legacy);
  BENCH(slerp_legacy_nlerp);
  BENCH(slerp_rw);
  BENCH(slerp_gael);
}

