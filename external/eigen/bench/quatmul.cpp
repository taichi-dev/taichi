#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <bench/BenchTimer.h>

using namespace Eigen; 

template<typename Quat>
EIGEN_DONT_INLINE void quatmul_default(const Quat& a, const Quat& b, Quat& c)
{
  c = a * b;
}

template<typename Quat>
EIGEN_DONT_INLINE void quatmul_novec(const Quat& a, const Quat& b, Quat& c)
{
  c = internal::quat_product<0, Quat, Quat, typename Quat::Scalar, Aligned>::run(a,b);
}

template<typename Quat> void bench(const std::string& label)
{
  int tries = 10;
  int rep = 1000000;
  BenchTimer t;
  
  Quat a(4, 1, 2, 3);
  Quat b(2, 3, 4, 5);
  Quat c;
  
  std::cout.precision(3);
  
  BENCH(t, tries, rep, quatmul_default(a,b,c));
  std::cout << label << " default " << 1e3*t.best(CPU_TIMER) << "ms  \t" << 1e-6*double(rep)/(t.best(CPU_TIMER)) << " M mul/s\n";
  
  BENCH(t, tries, rep, quatmul_novec(a,b,c));
  std::cout << label << " novec   " << 1e3*t.best(CPU_TIMER) << "ms  \t" << 1e-6*double(rep)/(t.best(CPU_TIMER)) << " M mul/s\n";
}

int main()
{
  bench<Quaternionf>("float ");
  bench<Quaterniond>("double");

  return 0;

}

