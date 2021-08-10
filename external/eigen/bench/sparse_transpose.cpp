
//g++ -O3 -g0 -DNDEBUG  sparse_transpose.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.005 -DSIZE=10000 && ./a.out
// -DNOGMM -DNOMTL
// -DCSPARSE -I /home/gael/Coding/LinearAlgebra/CSparse/Include/ /home/gael/Coding/LinearAlgebra/CSparse/Lib/libcsparse.a

#ifndef SIZE
#define SIZE 10000
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

int main(int argc, char *argv[])
{
  int rows = SIZE;
  int cols = SIZE;
  float density = DENSITY;

  EigenSparseMatrix sm1(rows,cols), sm3(rows,cols);

  BenchTimer timer;
  for (float density = DENSITY; density>=MINDENSITY; density*=0.5)
  {
    fillMatrix(density, rows, cols, sm1);

    // dense matrices
    #ifdef DENSEMATRIX
    {
      DenseMatrix m1(rows,cols), m3(rows,cols);
      eiToDense(sm1, m1);
      BENCH(for (int k=0; k<REPEAT; ++k) m3 = m1.transpose();)
      std::cout << "  Eigen dense:\t" << timer.value() << endl;
    }
    #endif

    std::cout << "Non zeros: " << sm1.nonZeros()/float(sm1.rows()*sm1.cols())*100 << "%\n";

    // eigen sparse matrices
    {
      BENCH(for (int k=0; k<REPEAT; ++k) sm3 = sm1.transpose();)
      std::cout << "  Eigen:\t" << timer.value() << endl;
    }

    // CSparse
    #ifdef CSPARSE
    {
      cs *m1, *m3;
      eiToCSparse(sm1, m1);

      BENCH(for (int k=0; k<REPEAT; ++k) { m3 = cs_transpose(m1,1); cs_spfree(m3);})
      std::cout << "  CSparse:\t" << timer.value() << endl;
    }
    #endif

    // GMM++
    #ifndef NOGMM
    {
      GmmDynSparse  gmmT3(rows,cols);
      GmmSparse m1(rows,cols), m3(rows,cols);
      eiToGmm(sm1, m1);
      BENCH(for (int k=0; k<REPEAT; ++k) gmm::copy(gmm::transposed(m1),m3);)
      std::cout << "  GMM:\t\t" << timer.value() << endl;
    }
    #endif

    // MTL4
    #ifndef NOMTL
    {
      MtlSparse m1(rows,cols), m3(rows,cols);
      eiToMtl(sm1, m1);
      BENCH(for (int k=0; k<REPEAT; ++k) m3 = trans(m1);)
      std::cout << "  MTL4:\t\t" << timer.value() << endl;
    }
    #endif

    std::cout << "\n\n";
  }

  return 0;
}

