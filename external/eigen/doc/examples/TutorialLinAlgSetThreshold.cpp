#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
   Matrix2d A;
   A << 2, 1,
        2, 0.9999999999;
   FullPivLU<Matrix2d> lu(A);
   cout << "By default, the rank of A is found to be " << lu.rank() << endl;
   lu.setThreshold(1e-5);
   cout << "With threshold 1e-5, the rank of A is found to be " << lu.rank() << endl;
}
