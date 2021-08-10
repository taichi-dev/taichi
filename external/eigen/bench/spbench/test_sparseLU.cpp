// Small bench routine for Eigen available in Eigen
// (C) Desire NUENTSA WAKAM, INRIA

#include <iostream>
#include <fstream>
#include <iomanip>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseLU>
#include <bench/BenchTimer.h>
#ifdef EIGEN_METIS_SUPPORT
#include <Eigen/MetisSupport>
#endif

using namespace std;
using namespace Eigen;

int main(int argc, char **args)
{
//   typedef complex<double> scalar; 
  typedef double scalar; 
  SparseMatrix<scalar, ColMajor> A; 
  typedef SparseMatrix<scalar, ColMajor>::Index Index;
  typedef Matrix<scalar, Dynamic, Dynamic> DenseMatrix;
  typedef Matrix<scalar, Dynamic, 1> DenseRhs;
  Matrix<scalar, Dynamic, 1> b, x, tmp;
//   SparseLU<SparseMatrix<scalar, ColMajor>, AMDOrdering<int> >   solver;
// #ifdef EIGEN_METIS_SUPPORT
//   SparseLU<SparseMatrix<scalar, ColMajor>, MetisOrdering<int> > solver; 
//   std::cout<< "ORDERING : METIS\n"; 
// #else
  SparseLU<SparseMatrix<scalar, ColMajor>, COLAMDOrdering<int> >  solver;
  std::cout<< "ORDERING : COLAMD\n"; 
// #endif
  
  ifstream matrix_file; 
  string line;
  int  n;
  BenchTimer timer; 
  
  // Set parameters
  /* Fill the matrix with sparse matrix stored in Matrix-Market coordinate column-oriented format */
  if (argc < 2) assert(false && "please, give the matrix market file ");
  loadMarket(A, args[1]);
  cout << "End charging matrix " << endl;
  bool iscomplex=false, isvector=false;
  int sym;
  getMarketHeader(args[1], sym, iscomplex, isvector);
//   if (iscomplex) { cout<< " Not for complex matrices \n"; return -1; }
  if (isvector) { cout << "The provided file is not a matrix file\n"; return -1;}
  if (sym != 0) { // symmetric matrices, only the lower part is stored
    SparseMatrix<scalar, ColMajor> temp; 
    temp = A;
    A = temp.selfadjointView<Lower>();
  }
  n = A.cols();
  /* Fill the right hand side */

  if (argc > 2)
    loadMarketVector(b, args[2]);
  else 
  {
    b.resize(n);
    tmp.resize(n);
//       tmp.setRandom();
    for (int i = 0; i < n; i++) tmp(i) = i; 
    b = A * tmp ;
  }

  /* Compute the factorization */
//   solver.isSymmetric(true);
  timer.start(); 
//   solver.compute(A);
  solver.analyzePattern(A); 
  timer.stop(); 
  cout << "Time to analyze " << timer.value() << std::endl;
  timer.reset(); 
  timer.start(); 
  solver.factorize(A); 
  timer.stop(); 
  cout << "Factorize Time " << timer.value() << std::endl;
  timer.reset(); 
  timer.start(); 
  x = solver.solve(b);
  timer.stop();
  cout << "solve time " << timer.value() << std::endl; 
  /* Check the accuracy */
  Matrix<scalar, Dynamic, 1> tmp2 = b - A*x;
  scalar tempNorm = tmp2.norm()/b.norm();
  cout << "Relative norm of the computed solution : " << tempNorm <<"\n";
  cout << "Number of nonzeros in the factor : " << solver.nnzL() + solver.nnzU() << std::endl; 
  
  return 0;
}