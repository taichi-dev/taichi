#include <iostream>
struct init {
  init() { std::cout << "[" << "init" << "]" << std::endl; }
};
init init_obj;
// [init]
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
  MatrixXd A(2,2);
  A << 2, -1, 1, 3;
  cout << "Here is the input matrix A before decomposition:\n" << A << endl;
cout << "[init]" << endl;

cout << "[declaration]" << endl;
  PartialPivLU<Ref<MatrixXd> > lu(A);
  cout << "Here is the input matrix A after decomposition:\n" << A << endl;
cout << "[declaration]" << endl;

cout << "[matrixLU]" << endl;
  cout << "Here is the matrix storing the L and U factors:\n" << lu.matrixLU() << endl;
cout << "[matrixLU]" << endl;

cout << "[solve]" << endl;
  MatrixXd A0(2,2); A0 << 2, -1, 1, 3;
  VectorXd b(2);    b << 1, 2;
  VectorXd x = lu.solve(b);
  cout << "Residual: " << (A0 * x - b).norm() << endl;
cout << "[solve]" << endl;

cout << "[modifyA]" << endl;
  A << 3, 4, -2, 1;
  x = lu.solve(b);
  cout << "Residual: " << (A0 * x - b).norm() << endl;
cout << "[modifyA]" << endl;

cout << "[recompute]" << endl;
  A0 = A; // save A
  lu.compute(A);
  x = lu.solve(b);
  cout << "Residual: " << (A0 * x - b).norm() << endl;
cout << "[recompute]" << endl;

cout << "[recompute_bis0]" << endl;
  MatrixXd A1(2,2);
  A1 << 5,-2,3,4;
  lu.compute(A1);
  cout << "Here is the input matrix A1 after decomposition:\n" << A1 << endl;
cout << "[recompute_bis0]" << endl;

cout << "[recompute_bis1]" << endl;
  x = lu.solve(b);
  cout << "Residual: " << (A1 * x - b).norm() << endl;
cout << "[recompute_bis1]" << endl;

}
