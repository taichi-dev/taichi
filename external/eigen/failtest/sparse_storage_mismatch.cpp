#include "../Eigen/Sparse"
using namespace Eigen;

typedef SparseMatrix<double,ColMajor> Mat1;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
typedef SparseMatrix<double,RowMajor> Mat2;
#else
typedef SparseMatrix<double,ColMajor> Mat2;
#endif

int main()
{
  Mat1 a(10,10);
  Mat2 b(10,10);
  a += b;
}
