#include <Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace std;

int main(void)
{
    int const N = 5;
    MatrixXi A(N,N);
    A.setRandom();
    cout << "A =\n" << A << '\n' << endl;
    cout << "A(1..3,:) =\n" << A.middleCols(1,3) << endl;
    return 0;
}
