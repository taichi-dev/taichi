#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
   Matrix3f A;
   A << 1, 2, 5,
        2, 1, 4,
        3, 0, 3;
   cout << "Here is the matrix A:\n" << A << endl;
   FullPivLU<Matrix3f> lu_decomp(A);
   cout << "The rank of A is " << lu_decomp.rank() << endl;
   cout << "Here is a matrix whose columns form a basis of the null-space of A:\n"
        << lu_decomp.kernel() << endl;
   cout << "Here is a matrix whose columns form a basis of the column-space of A:\n"
        << lu_decomp.image(A) << endl; // yes, have to pass the original A
}
