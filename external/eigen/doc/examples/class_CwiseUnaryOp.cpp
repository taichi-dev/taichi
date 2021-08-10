#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;

// define a custom template unary functor
template<typename Scalar>
struct CwiseClampOp {
  CwiseClampOp(const Scalar& inf, const Scalar& sup) : m_inf(inf), m_sup(sup) {}
  const Scalar operator()(const Scalar& x) const { return x<m_inf ? m_inf : (x>m_sup ? m_sup : x); }
  Scalar m_inf, m_sup;
};

int main(int, char**)
{
  Matrix4d m1 = Matrix4d::Random();
  cout << m1 << endl << "becomes: " << endl << m1.unaryExpr(CwiseClampOp<double>(-0.5,0.5)) << endl;
  return 0;
}
