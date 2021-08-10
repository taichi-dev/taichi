#include <Eigen/Core>
#include <iostream>

using namespace Eigen;

int main()
{
  for (int size=1; size<=4; ++size)
  {
    MatrixXi m(size,size+1);         // a (size)x(size+1)-matrix of int's
    for (int j=0; j<m.cols(); ++j)   // loop over columns
      for (int i=0; i<m.rows(); ++i) // loop over rows
        m(i,j) = i+j*size;           // to access matrix coefficients,
                                     // use operator()(int,int)
    std::cout << m << "\n\n";
  }

  VectorXf v(4); // a vector of 4 float's
  // to access vector coefficients, use either operator () or operator []
  v[0] = 1; v[1] = 2; v(2) = 3; v(3) = 4;
  std::cout << "\nv:\n" << v << std::endl;
}
