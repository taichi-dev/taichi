#include "../Eigen/Core"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define CV_QUALIFIER
#else
#define CV_QUALIFIER const
#endif

using namespace Eigen;

void foo(const float *ptr, DenseIndex rows, DenseIndex cols){
    Map<CV_QUALIFIER MatrixXf, Unaligned, OuterStride<> > m(ptr, rows, cols, OuterStride<>(2));
}

int main() {}
