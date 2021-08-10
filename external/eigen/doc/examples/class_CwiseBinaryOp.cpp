#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;

// define a custom template binary functor
template<typename Scalar> struct MakeComplexOp {
  EIGEN_EMPTY_STRUCT_CTOR(MakeComplexOp)
  typedef complex<Scalar> result_type;
  complex<Scalar> operator()(const Scalar& a, const Scalar& b) const { return complex<Scalar>(a,b); }
};

int main(int, char**)
{
  Matrix4d m1 = Matrix4d::Random(), m2 = Matrix4d::Random();
  cout << m1.binaryExpr(m2, MakeComplexOp<double>()) << endl;
  return 0;
}
