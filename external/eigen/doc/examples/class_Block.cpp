#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;

template<typename Derived>
Eigen::Block<Derived>
topLeftCorner(MatrixBase<Derived>& m, int rows, int cols)
{
  return Eigen::Block<Derived>(m.derived(), 0, 0, rows, cols);
}

template<typename Derived>
const Eigen::Block<const Derived>
topLeftCorner(const MatrixBase<Derived>& m, int rows, int cols)
{
  return Eigen::Block<const Derived>(m.derived(), 0, 0, rows, cols);
}

int main(int, char**)
{
  Matrix4d m = Matrix4d::Identity();
  cout << topLeftCorner(4*m, 2, 3) << endl; // calls the const version
  topLeftCorner(m, 2, 3) *= 5;              // calls the non-const version
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
