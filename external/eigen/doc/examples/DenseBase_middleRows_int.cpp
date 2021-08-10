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
    cout << "A(2..3,:) =\n" << A.middleRows(2,2) << endl;
    return 0;
}
