#include "../Eigen/Sparse"

using namespace Eigen;

void call_ref(Ref<SparseMatrix<float> > a) { }

int main()
{
  SparseMatrix<float> A(10,10);
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  call_ref(A.row(3));
#else
  call_ref(A.col(3));
#endif
}
