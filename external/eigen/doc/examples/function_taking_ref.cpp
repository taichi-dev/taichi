#include <iostream>
#include <Eigen/SVD>
using namespace Eigen;
using namespace std;

float inv_cond(const Ref<const MatrixXf>& a)
{
  const VectorXf sing_vals = a.jacobiSvd().singularValues();
  return sing_vals(sing_vals.size()-1) / sing_vals(0);
}

int main()
{
  Matrix4f m = Matrix4f::Random();
  cout << "matrix m:" << endl << m << endl << endl;
  cout << "inv_cond(m):          " << inv_cond(m)                      << endl;
  cout << "inv_cond(m(1:3,1:3)): " << inv_cond(m.topLeftCorner(3,3))   << endl;
  cout << "inv_cond(m+I):        " << inv_cond(m+Matrix4f::Identity()) << endl;
}
