#include "../Eigen/Core"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define CV_QUALIFIER const
#else
#define CV_QUALIFIER
#endif

using namespace Eigen;

void call_ref(Ref<VectorXf> a) { }

int main()
{
  VectorXf a(10);
  CV_QUALIFIER VectorXf& ac(a);
  call_ref(ac);
}
