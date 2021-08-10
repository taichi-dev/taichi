#include "../Eigen/Core"

using namespace Eigen;

void call_ref(Ref<MatrixXf,0,OuterStride<> > a) {}

int main()
{
  MatrixXf A(10,10);
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  call_ref(A.transpose());
#else
  call_ref(A);
#endif
}
