// Small bench routine for Eigen available in Eigen
// (C) Desire NUENTSA WAKAM, INRIA

#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Jacobi>
#include <Eigen/Householder>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include <unsupported/Eigen/SparseExtra>
//#include <Eigen/SparseLU>
#include <Eigen/SuperLUSupport>
// #include <unsupported/Eigen/src/IterativeSolvers/Scaling.h>
#include <bench/BenchTimer.h>
#include <unsupported/Eigen/IterativeSolvers>
using namespace std;
using namespace Eigen;

int main(int argc, char **args)
{
  SparseMatrix<double, ColMajor> A; 
  typedef SparseMatrix<double, ColMajor>::Index Index;
  typedef Matrix<double, Dynamic, Dynamic> DenseMatrix;
  typedef Matrix<double, Dynamic, 1> DenseRhs;
  VectorXd b, x, tmp;
  BenchTimer timer,totaltime; 
  //SparseLU<SparseMatrix<double, ColMajor> >   solver;
//   SuperLU<SparseMatrix<double, ColMajor> >   solver;
  ConjugateGradient<SparseMatrix<double, ColMajor>, Lower,IncompleteCholesky<double,Lower> > solver; 
  ifstream matrix_file; 
  string line;
  int  n;
  // Set parameters
//   solver.iparm(IPARM_THREAD_NBR) = 4;
  /* Fill the matrix with sparse matrix stored in Matrix-Market coordinate column-oriented format */
  if (argc < 2) assert(false && "please, give the matrix market file ");
  
  timer.start();
  totaltime.start();
  loadMarket(A, args[1]);
  cout << "End charging matrix " << endl;
  bool iscomplex=false, isvector=false;
  int sym;
  getMarketHeader(args[1], sym, iscomplex, isvector);
  if (iscomplex) { cout<< " Not for complex matrices \n"; return -1; }
  if (isvector) { cout << "The provided file is not a matrix file\n"; return -1;}
  if (sym != 0) { // symmetric matrices, only the lower part is stored
    SparseMatrix<double, ColMajor> temp; 
    temp = A;
    A = temp.selfadjointView<Lower>();
  }
  timer.stop();
  
  n = A.cols();
  // ====== TESTS FOR SPARSE TUTORIAL ======
//   cout<< "OuterSize " << A.outerSize() << " inner " << A.innerSize() << endl; 
//   SparseMatrix<double, RowMajor> mat1(A); 
//   SparseMatrix<double, RowMajor> mat2;
//   cout << " norm of A " << mat1.norm() << endl; ;
//   PermutationMatrix<Dynamic, Dynamic, int> perm(n);
//   perm.resize(n,1);
//   perm.indices().setLinSpaced(n, 0, n-1);
//   mat2 = perm * mat1;
//   mat.subrows();
//   mat2.resize(n,n); 
//   mat2.reserve(10);
//   mat2.setConstant();
//   std::cout<< "NORM " << mat1.squaredNorm()<< endl;  

  cout<< "Time to load the matrix " << timer.value() <<endl;
  /* Fill the right hand side */

//   solver.set_restart(374);
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
//   Scaling<SparseMatrix<double> > scal; 
//   scal.computeRef(A);
//   b = scal.LeftScaling().cwiseProduct(b);

  /* Compute the factorization */
  cout<< "Starting the factorization "<< endl; 
  timer.reset();
  timer.start(); 
  cout<< "Size of Input Matrix "<< b.size()<<"\n\n";
  cout<< "Rows and columns "<< A.rows() <<" " <<A.cols() <<"\n";
  solver.compute(A);
//   solver.analyzePattern(A);
//   solver.factorize(A);
  if (solver.info() != Success) {
    std::cout<< "The solver failed \n";
    return -1; 
  }
  timer.stop(); 
  float time_comp = timer.value(); 
  cout <<" Compute Time " << time_comp<< endl; 
  
  timer.reset();
  timer.start();
  x = solver.solve(b);
//   x = scal.RightScaling().cwiseProduct(x);
  timer.stop();
  float time_solve = timer.value(); 
  cout<< " Time to solve " << time_solve << endl; 
 
  /* Check the accuracy */
  VectorXd tmp2 = b - A*x;
  double tempNorm = tmp2.norm()/b.norm();
  cout << "Relative norm of the computed solution : " << tempNorm <<"\n";
//   cout << "Iterations : " << solver.iterations() << "\n"; 
  
  totaltime.stop();
  cout << "Total time " << totaltime.value() << "\n";
//  std::cout<<x.transpose()<<"\n";
  
  return 0;
}