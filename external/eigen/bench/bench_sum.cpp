#include <iostream>
#include <Eigen/Core>
using namespace Eigen;
using namespace std;

int main() 
{
  typedef Matrix<SCALAR,Eigen::Dynamic,1> Vec;
  Vec v(SIZE);
  v.setZero();
  v[0] = 1;
  v[1] = 2;
  for(int i = 0; i < 1000000; i++)
  {
    v.coeffRef(0) += v.sum() * SCALAR(1e-20);
  }
  cout << v.sum() << endl;
}
