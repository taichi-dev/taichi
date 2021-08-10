
// g++ -I.. sparse_lu.cpp -O3 -g0 -I /usr/include/superlu/ -lsuperlu -lgfortran -DSIZE=1000 -DDENSITY=.05 && ./a.out

#define EIGEN_SUPERLU_SUPPORT
#define EIGEN_UMFPACK_SUPPORT
#include <Eigen/Sparse>

#define NOGMM
#define NOMTL

#ifndef SIZE
#define SIZE 10
#endif

#ifndef DENSITY
#define DENSITY 0.01
#endif

#ifndef REPEAT
#define REPEAT 1
#endif

#include "BenchSparseUtil.h"

#ifndef MINDENSITY
#define MINDENSITY 0.0004
#endif

#ifndef NBTRIES
#define NBTRIES 10
#endif

#define BENCH(X) \
  timer.reset(); \
  for (int _j=0; _j<NBTRIES; ++_j) { \
    timer.start(); \
    for (int _k=0; _k<REPEAT; ++_k) { \
        X  \
  } timer.stop(); }

typedef Matrix<Scalar,Dynamic,1> VectorX;

#include <Eigen/LU>

template<int Backend>
void doEigen(const char* name, const EigenSparseMatrix& sm1, const VectorX& b, VectorX& x, int flags = 0)
{
  std::cout << name << "..." << std::flush;
  BenchTimer timer; timer.start();
  SparseLU<EigenSparseMatrix,Backend> lu(sm1, flags);
  timer.stop();
  if (lu.succeeded())
    std::cout << ":\t" << timer.value() << endl;
  else
  {
    std::cout << ":\t FAILED" << endl;
    return;
  }

  bool ok;
  timer.reset(); timer.start();
  ok = lu.solve(b,&x);
  timer.stop();
  if (ok)
    std::cout << "  solve:\t" << timer.value() << endl;
  else
    std::cout << "  solve:\t" << " FAILED" << endl;

  //std::cout << x.transpose() << "\n";
}

int main(int argc, char *argv[])
{
  int rows = SIZE;
  int cols = SIZE;
  float density = DENSITY;
  BenchTimer timer;

  VectorX b = VectorX::Random(cols);
  VectorX x = VectorX::Random(cols);

  bool densedone = false;

  //for (float density = DENSITY; density>=MINDENSITY; density*=0.5)
//   float density = 0.5;
  {
    EigenSparseMatrix sm1(rows, cols);
    fillMatrix(density, rows, cols, sm1);

    // dense matrices
    #ifdef DENSEMATRIX
    if (!densedone)
    {
      densedone = true;
      std::cout << "Eigen Dense\t" << density*100 << "%\n";
      DenseMatrix m1(rows,cols);
      eiToDense(sm1, m1);

      BenchTimer timer;
      timer.start();
      FullPivLU<DenseMatrix> lu(m1);
      timer.stop();
      std::cout << "Eigen/dense:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      lu.solve(b,&x);
      timer.stop();
      std::cout << "  solve:\t" << timer.value() << endl;
//       std::cout << b.transpose() << "\n";
//       std::cout << x.transpose() << "\n";
    }
    #endif

    #ifdef EIGEN_UMFPACK_SUPPORT
    x.setZero();
    doEigen<Eigen::UmfPack>("Eigen/UmfPack (auto)", sm1, b, x, 0);
    #endif

    #ifdef EIGEN_SUPERLU_SUPPORT
    x.setZero();
    doEigen<Eigen::SuperLU>("Eigen/SuperLU (nat)", sm1, b, x, Eigen::NaturalOrdering);
//     doEigen<Eigen::SuperLU>("Eigen/SuperLU (MD AT+A)", sm1, b, x, Eigen::MinimumDegree_AT_PLUS_A);
//     doEigen<Eigen::SuperLU>("Eigen/SuperLU (MD ATA)", sm1, b, x, Eigen::MinimumDegree_ATA);
    doEigen<Eigen::SuperLU>("Eigen/SuperLU (COLAMD)", sm1, b, x, Eigen::ColApproxMinimumDegree);
    #endif

  }

  return 0;
}

