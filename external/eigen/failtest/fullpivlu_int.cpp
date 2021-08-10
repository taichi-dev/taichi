#include "../Eigen/LU"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define SCALAR int
#else
#define SCALAR float
#endif

using namespace Eigen;

int main()
{
  FullPivLU<Matrix<SCALAR,Dynamic,Dynamic> > lu(Matrix<SCALAR,Dynamic,Dynamic>::Random(10,10));
}
