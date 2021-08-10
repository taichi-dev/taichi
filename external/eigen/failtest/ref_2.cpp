#include "../Eigen/Core"

using namespace Eigen;

void call_ref(Ref<VectorXf> a) { }

int main()
{
  MatrixXf A(10,10);
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  call_ref(A.row(3));
#else
  call_ref(A.col(3));
#endif
}
