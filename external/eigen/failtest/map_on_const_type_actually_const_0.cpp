#include "../Eigen/Core"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define CV_QUALIFIER const
#else
#define CV_QUALIFIER
#endif

using namespace Eigen;

void foo(float *ptr){
    Map<CV_QUALIFIER MatrixXf>(ptr, 1, 1).coeffRef(0,0) = 1.0f;
}

int main() {}
