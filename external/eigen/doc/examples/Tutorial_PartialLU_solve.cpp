#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
   Matrix3f A;
   Vector3f b;
   A << 1,2,3,  4,5,6,  7,8,10;
   b << 3, 3, 4;
   cout << "Here is the matrix A:" << endl << A << endl;
   cout << "Here is the vector b:" << endl << b << endl;
   Vector3f x = A.lu().solve(b);
   cout << "The solution is:" << endl << x << endl;
}
