
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.005 -DSIZE=10000 && ./a.out
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.05 -DSIZE=2000 && ./a.out
// -DNOGMM -DNOMTL
// -I /home/gael/Coding/LinearAlgebra/CSparse/Include/ /home/gael/Coding/LinearAlgebra/CSparse/Lib/libcsparse.a

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

typedef SparseMatrix<Scalar,UpperTriangular> EigenSparseTriMatrix;
typedef SparseMatrix<Scalar,RowMajorBit|UpperTriangular> EigenSparseTriMatrixRow;

void fillMatrix(float density, int rows, int cols,  EigenSparseTriMatrix& dst)
{
  dst.startFill(rows*cols*density);
  for(int j = 0; j < cols; j++)
  {
    for(int i = 0; i < j; i++)
    {
      Scalar v = (internal::random<float>(0,1) < density) ? internal::random<Scalar>() : 0;
      if (v!=0)
        dst.fill(i,j) = v;
    }
    dst.fill(j,j) = internal::random<Scalar>();
  }
  dst.endFill();
}

int main(int argc, char *argv[])
{
  int rows = SIZE;
  int cols = SIZE;
  float density = DENSITY;
  BenchTimer timer;
  #if 1
  EigenSparseTriMatrix sm1(rows,cols);
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  DenseVector b = DenseVector::Random(cols);
  DenseVector x = DenseVector::Random(cols);

  bool densedone = false;

  for (float density = DENSITY; density>=MINDENSITY; density*=0.5)
  {
    EigenSparseTriMatrix sm1(rows, cols);
    fillMatrix(density, rows, cols, sm1);

    // dense matrices
    #ifdef DENSEMATRIX
    if (!densedone)
    {
      densedone = true;
      std::cout << "Eigen Dense\t" << density*100 << "%\n";
      DenseMatrix m1(rows,cols);
      Matrix<Scalar,Dynamic,Dynamic,Dynamic,Dynamic,RowMajorBit> m2(rows,cols);
      eiToDense(sm1, m1);
      m2 = m1;

      BENCH(x = m1.marked<UpperTriangular>().solveTriangular(b);)
      std::cout << "   colmajor^-1 * b:\t" << timer.value() << endl;
//       std::cerr << x.transpose() << "\n";

      BENCH(x = m2.marked<UpperTriangular>().solveTriangular(b);)
      std::cout << "   rowmajor^-1 * b:\t" << timer.value() << endl;
//       std::cerr << x.transpose() << "\n";
    }
    #endif

    // eigen sparse matrices
    {
      std::cout << "Eigen sparse\t" << density*100 << "%\n";
      EigenSparseTriMatrixRow sm2 = sm1;

      BENCH(x = sm1.solveTriangular(b);)
      std::cout << "   colmajor^-1 * b:\t" << timer.value() << endl;
//       std::cerr << x.transpose() << "\n";

      BENCH(x = sm2.solveTriangular(b);)
      std::cout << "   rowmajor^-1 * b:\t" << timer.value() << endl;
//       std::cerr << x.transpose() << "\n";

//       x = b;
//       BENCH(sm1.inverseProductInPlace(x);)
//       std::cout << "   colmajor^-1 * b:\t" << timer.value() << " (inplace)" << endl;
//       std::cerr << x.transpose() << "\n";
//
//       x = b;
//       BENCH(sm2.inverseProductInPlace(x);)
//       std::cout << "   rowmajor^-1 * b:\t" << timer.value() << " (inplace)" << endl;
//       std::cerr << x.transpose() << "\n";
    }



    // CSparse
    #ifdef CSPARSE
    {
      std::cout << "CSparse \t" << density*100 << "%\n";
      cs *m1;
      eiToCSparse(sm1, m1);

      BENCH(x = b; if (!cs_lsolve (m1, x.data())){std::cerr << "cs_lsolve failed\n"; break;}; )
      std::cout << "   colmajor^-1 * b:\t" << timer.value() << endl;
    }
    #endif

    // GMM++
    #ifndef NOGMM
    {
      std::cout << "GMM++ sparse\t" << density*100 << "%\n";
      GmmSparse m1(rows,cols);
      gmm::csr_matrix<Scalar> m2;
      eiToGmm(sm1, m1);
      gmm::copy(m1,m2);
      std::vector<Scalar> gmmX(cols), gmmB(cols);
      Map<Matrix<Scalar,Dynamic,1> >(&gmmX[0], cols) = x;
      Map<Matrix<Scalar,Dynamic,1> >(&gmmB[0], cols) = b;

      gmmX = gmmB;
      BENCH(gmm::upper_tri_solve(m1, gmmX, false);)
      std::cout << "   colmajor^-1 * b:\t" << timer.value() << endl;
//       std::cerr << Map<Matrix<Scalar,Dynamic,1> >(&gmmX[0], cols).transpose() << "\n";

      gmmX = gmmB;
      BENCH(gmm::upper_tri_solve(m2, gmmX, false);)
      timer.stop();
      std::cout << "   rowmajor^-1 * b:\t" << timer.value() << endl;
//       std::cerr << Map<Matrix<Scalar,Dynamic,1> >(&gmmX[0], cols).transpose() << "\n";
    }
    #endif

    // MTL4
    #ifndef NOMTL
    {
      std::cout << "MTL4\t" << density*100 << "%\n";
      MtlSparse m1(rows,cols);
      MtlSparseRowMajor m2(rows,cols);
      eiToMtl(sm1, m1);
      m2 = m1;
      mtl::dense_vector<Scalar> x(rows, 1.0);
      mtl::dense_vector<Scalar> b(rows, 1.0);

      BENCH(x = mtl::upper_trisolve(m1,b);)
      std::cout << "   colmajor^-1 * b:\t" << timer.value() << endl;
//       std::cerr << x << "\n";

      BENCH(x = mtl::upper_trisolve(m2,b);)
      std::cout << "   rowmajor^-1 * b:\t" << timer.value() << endl;
//       std::cerr << x << "\n";
    }
    #endif


    std::cout << "\n\n";
  }
  #endif

  #if 0
    // bench small matrices (in-place versus return bye value)
    {
      timer.reset();
      for (int _j=0; _j<10; ++_j) {
        Matrix4f m = Matrix4f::Random();
        Vector4f b = Vector4f::Random();
        Vector4f x = Vector4f::Random();
        timer.start();
        for (int _k=0; _k<1000000; ++_k) {
          b = m.inverseProduct(b);
        }
        timer.stop();
      }
      std::cout << "4x4 :\t" << timer.value() << endl;
    }

    {
      timer.reset();
      for (int _j=0; _j<10; ++_j) {
        Matrix4f m = Matrix4f::Random();
        Vector4f b = Vector4f::Random();
        Vector4f x = Vector4f::Random();
        timer.start();
        for (int _k=0; _k<1000000; ++_k) {
          m.inverseProductInPlace(x);
        }
        timer.stop();
      }
      std::cout << "4x4 IP :\t" << timer.value() << endl;
    }
  #endif

  return 0;
}

