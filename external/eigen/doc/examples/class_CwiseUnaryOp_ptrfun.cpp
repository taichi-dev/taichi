#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;

// define function to be applied coefficient-wise
double ramp(double x)
{
  if (x > 0)
    return x;
  else 
    return 0;
}

int main(int, char**)
{
  Matrix4d m1 = Matrix4d::Random();
  cout << m1 << endl << "becomes: " << endl << m1.unaryExpr(ptr_fun(ramp)) << endl;
  return 0;
}
