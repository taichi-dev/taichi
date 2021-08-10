#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;

template<typename Derived>
Eigen::VectorBlock<Derived, 2>
firstTwo(MatrixBase<Derived>& v)
{
  return Eigen::VectorBlock<Derived, 2>(v.derived(), 0);
}

template<typename Derived>
const Eigen::VectorBlock<const Derived, 2>
firstTwo(const MatrixBase<Derived>& v)
{
  return Eigen::VectorBlock<const Derived, 2>(v.derived(), 0);
}

int main(int, char**)
{
  Matrix<int,1,6> v; v << 1,2,3,4,5,6;
  cout << firstTwo(4*v) << endl; // calls the const version
  firstTwo(v) *= 2;              // calls the non-const version
  cout << "Now the vector v is:" << endl << v << endl;
  return 0;
}
