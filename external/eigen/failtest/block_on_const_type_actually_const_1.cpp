#include "../Eigen/Core"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define CV_QUALIFIER const
#else
#define CV_QUALIFIER
#endif

using namespace Eigen;

void foo(){
    MatrixXf m;
    Block<CV_QUALIFIER MatrixXf, 3, 3>(m, 0, 0).coeffRef(0, 0) = 1.0f;
}

int main() {}
