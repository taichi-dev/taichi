#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  ArrayXXf a(2,2);
  
  a << 1,2,
       3,4;

  cout << "(a > 0).all()   = " << (a > 0).all() << endl;
  cout << "(a > 0).any()   = " << (a > 0).any() << endl;
  cout << "(a > 0).count() = " << (a > 0).count() << endl;
  cout << endl;
  cout << "(a > 2).all()   = " << (a > 2).all() << endl;
  cout << "(a > 2).any()   = " << (a > 2).any() << endl;
  cout << "(a > 2).count() = " << (a > 2).count() << endl;
}
