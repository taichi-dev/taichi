#include "../Eigen/Core"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define CV_QUALIFIER const
#else
#define CV_QUALIFIER
#endif

using namespace Eigen;

void foo(CV_QUALIFIER float *ptr, DenseIndex rows, DenseIndex cols){
    Map<MatrixXf, Aligned, InnerStride<2> > m(ptr, rows, cols, InnerStride<2>());
}

int main() {}
