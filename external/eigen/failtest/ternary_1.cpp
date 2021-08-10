#include "../Eigen/Core"

using namespace Eigen;

int main(int argc,char **)
{
  VectorXf a(10), b(10);
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  b = argc>1 ? 2*a : -a;
#else
  b = argc>1 ? 2*a : VectorXf(-a);
#endif
}
