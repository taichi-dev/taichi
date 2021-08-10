#include "../Eigen/Cholesky"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define SCALAR int
#else
#define SCALAR float
#endif

using namespace Eigen;

int main()
{
  LLT<Matrix<SCALAR,Dynamic,Dynamic> > llt(Matrix<SCALAR,Dynamic,Dynamic>::Random(10,10));
}
