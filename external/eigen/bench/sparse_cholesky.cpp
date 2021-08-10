// #define EIGEN_TAUCS_SUPPORT
// #define EIGEN_CHOLMOD_SUPPORT
#include <iostream>
#include <Eigen/Sparse>

// g++ -DSIZE=10000 -DDENSITY=0.001  sparse_cholesky.cpp -I.. -DDENSEMATRI -O3 -g0 -DNDEBUG   -DNBTRIES=1 -I /home/gael/Coding/LinearAlgebra/taucs_full/src/ -I/home/gael/Coding/LinearAlgebra/taucs_full/build/linux/  -L/home/gael/Coding/LinearAlgebra/taucs_full/lib/linux/ -ltaucs /home/gael/Coding/LinearAlgebra/GotoBLAS/libgoto.a -lpthread -I /home/gael/Coding/LinearAlgebra/SuiteSparse/CHOLMOD/Include/ $CHOLLIB -I /home/gael/Coding/LinearAlgebra/SuiteSparse/UFconfig/ /home/gael/Coding/LinearAlgebra/SuiteSparse/CCOLAMD/Lib/libccolamd.a   /home/gael/Coding/LinearAlgebra/SuiteSparse/CHOLMOD/Lib/libcholmod.a -lmetis /home/gael/Coding/LinearAlgebra/SuiteSparse/AMD/Lib/libamd.a  /home/gael/Coding/LinearAlgebra/SuiteSparse/CAMD/Lib/libcamd.a   /home/gael/Coding/LinearAlgebra/SuiteSparse/CCOLAMD/Lib/libccolamd.a  /home/gael/Coding/LinearAlgebra/SuiteSparse/COLAMD/Lib/libcolamd.a -llapack && ./a.out

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

// typedef SparseMatrix<Scalar,UpperTriangular> EigenSparseTriMatrix;
typedef SparseMatrix<Scalar,SelfAdjoint|LowerTriangular> EigenSparseSelfAdjointMatrix;

void fillSpdMatrix(float density, int rows, int cols,  EigenSparseSelfAdjointMatrix& dst)
{
  dst.startFill(rows*cols*density);
  for(int j = 0; j < cols; j++)
  {
    dst.fill(j,j) = internal::random<Scalar>(10,20);
    for(int i = j+1; i < rows; i++)
    {
      Scalar v = (internal::random<float>(0,1) < density) ? internal::random<Scalar>() : 0;
      if (v!=0)
        dst.fill(i,j) = v;
    }

  }
  dst.endFill();
}

#include <Eigen/Cholesky>

template<int Backend>
void doEigen(const char* name, const EigenSparseSelfAdjointMatrix& sm1, int flags = 0)
{
  std::cout << name << "..." << std::flush;
  BenchTimer timer;
  timer.start();
  SparseLLT<EigenSparseSelfAdjointMatrix,Backend> chol(sm1, flags);
  timer.stop();
  std::cout << ":\t" << timer.value() << endl;

  std::cout << "  nnz: " << sm1.nonZeros() << " => " << chol.matrixL().nonZeros() << "\n";
//   std::cout << "sparse\n" << chol.matrixL() << "%\n";
}

int main(int argc, char *argv[])
{
  int rows = SIZE;
  int cols = SIZE;
  float density = DENSITY;
  BenchTimer timer;

  VectorXf b = VectorXf::Random(cols);
  VectorXf x = VectorXf::Random(cols);

  bool densedone = false;

  //for (float density = DENSITY; density>=MINDENSITY; density*=0.5)
//   float density = 0.5;
  {
    EigenSparseSelfAdjointMatrix sm1(rows, cols);
    std::cout << "Generate sparse matrix (might take a while)...\n";
    fillSpdMatrix(density, rows, cols, sm1);
    std::cout << "DONE\n\n";

    // dense matrices
    #ifdef DENSEMATRIX
    if (!densedone)
    {
      densedone = true;
      std::cout << "Eigen Dense\t" << density*100 << "%\n";
      DenseMatrix m1(rows,cols);
      eiToDense(sm1, m1);
      m1 = (m1 + m1.transpose()).eval();
      m1.diagonal() *= 0.5;

//       BENCH(LLT<DenseMatrix> chol(m1);)
//       std::cout << "dense:\t" << timer.value() << endl;

      BenchTimer timer;
      timer.start();
      LLT<DenseMatrix> chol(m1);
      timer.stop();
      std::cout << "dense:\t" << timer.value() << endl;
      int count = 0;
      for (int j=0; j<cols; ++j)
        for (int i=j; i<rows; ++i)
          if (!internal::isMuchSmallerThan(internal::abs(chol.matrixL()(i,j)), 0.1))
            count++;
      std::cout << "dense: " << "nnz = " << count << "\n";
//       std::cout << "dense:\n" << m1 << "\n\n" << chol.matrixL() << endl;
    }
    #endif

    // eigen sparse matrices
    doEigen<Eigen::DefaultBackend>("Eigen/Sparse", sm1, Eigen::IncompleteFactorization);

    #ifdef EIGEN_CHOLMOD_SUPPORT
    doEigen<Eigen::Cholmod>("Eigen/Cholmod", sm1, Eigen::IncompleteFactorization);
    #endif

    #ifdef EIGEN_TAUCS_SUPPORT
    doEigen<Eigen::Taucs>("Eigen/Taucs", sm1, Eigen::IncompleteFactorization);
    #endif

    #if 0
    // TAUCS
    {
      taucs_ccs_matrix A = sm1.asTaucsMatrix();

      //BENCH(taucs_ccs_matrix* chol = taucs_ccs_factor_llt(&A, 0, 0);)
//       BENCH(taucs_supernodal_factor_to_ccs(taucs_ccs_factor_llt_ll(&A));)
//       std::cout << "taucs:\t" << timer.value() << endl;

      taucs_ccs_matrix* chol = taucs_ccs_factor_llt(&A, 0, 0);

      for (int j=0; j<cols; ++j)
      {
        for (int i=chol->colptr[j]; i<chol->colptr[j+1]; ++i)
          std::cout << chol->values.d[i] << " ";
      }
    }

    // CHOLMOD
    #ifdef EIGEN_CHOLMOD_SUPPORT
    {
      cholmod_common c;
      cholmod_start (&c);
      cholmod_sparse A;
      cholmod_factor *L;

      A = sm1.asCholmodMatrix();
      BenchTimer timer;
//       timer.reset();
      timer.start();
      std::vector<int> perm(cols);
//       std::vector<int> set(ncols);
      for (int i=0; i<cols; ++i)
        perm[i] = i;
//       c.nmethods = 1;
//       c.method[0] = 1;

      c.nmethods = 1;
      c.method [0].ordering = CHOLMOD_NATURAL;
      c.postorder = 0;
      c.final_ll = 1;

      L = cholmod_analyze_p(&A, &perm[0], &perm[0], cols, &c);
      timer.stop();
      std::cout << "cholmod/analyze:\t" << timer.value() << endl;
      timer.reset();
      timer.start();
      cholmod_factorize(&A, L, &c);
      timer.stop();
      std::cout << "cholmod/factorize:\t" << timer.value() << endl;

      cholmod_sparse* cholmat = cholmod_factor_to_sparse(L, &c);

      cholmod_print_factor(L, "Factors", &c);

      cholmod_print_sparse(cholmat, "Chol", &c);
      cholmod_write_sparse(stdout, cholmat, 0, 0, &c);
//
//       cholmod_print_sparse(&A, "A", &c);
//       cholmod_write_sparse(stdout, &A, 0, 0, &c);


//       for (int j=0; j<cols; ++j)
//       {
//           for (int i=chol->colptr[j]; i<chol->colptr[j+1]; ++i)
//             std::cout << chol->values.s[i] << " ";
//       }
    }
    #endif

    #endif



  }


  return 0;
}

