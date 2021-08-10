#include "../Eigen/Sparse"

using namespace Eigen;

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
void call_ref(Ref<SparseMatrix<float> > a) { }
#else
void call_ref(const Ref<const SparseMatrix<float> > &a) { }
#endif

int main()
{
  SparseMatrix<float> a(10,10);
  call_ref(a+a);
}
