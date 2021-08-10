#include "../Eigen/Core"

using namespace Eigen;

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
void call_ref(Ref<VectorXf> a) { }
#else
void call_ref(const Ref<const VectorXf> &a) { }
#endif

int main()
{
  VectorXf a(10);
  call_ref(a+a);
}
