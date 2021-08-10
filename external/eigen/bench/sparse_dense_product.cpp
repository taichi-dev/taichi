
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.005 -DSIZE=10000 && ./a.out
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.05 -DSIZE=2000 && ./a.out
// -DNOGMM -DNOMTL -DCSPARSE
// -I /home/gael/Coding/LinearAlgebra/CSparse/Include/ /home/gael/Coding/LinearAlgebra/CSparse/Lib/libcsparse.a
#ifndef SIZE
#define SIZE 650000
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


#ifdef CSPARSE
cs* cs_sorted_multiply(const cs* a, const cs* b)
{
  cs* A = cs_transpose (a, 1) ;
  cs* B = cs_transpose (b, 1) ;
  cs* D = cs_multiply (B,A) ;   /* D = B'*A' */
  cs_spfree (A) ;
  cs_spfree (B) ;
  cs_dropzeros (D) ;      /* drop zeros from D */
  cs* C = cs_transpose (D, 1) ;   /* C = D', so that C is sorted */
  cs_spfree (D) ;
  return C;
}
#endif

int main(int argc, char *argv[])
{
  int rows = SIZE;
  int cols = SIZE;
  float density = DENSITY;

  EigenSparseMatrix sm1(rows,cols);
  DenseVector v1(cols), v2(cols);
  v1.setRandom();

  BenchTimer timer;
  for (float density = DENSITY; density>=MINDENSITY; density*=0.5)
  {
    //fillMatrix(density, rows, cols, sm1);
    fillMatrix2(7, rows, cols, sm1);

    // dense matrices
    #ifdef DENSEMATRIX
    {
      std::cout << "Eigen Dense\t" << density*100 << "%\n";
      DenseMatrix m1(rows,cols);
      eiToDense(sm1, m1);

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        v2 = m1 * v1;
      timer.stop();
      std::cout << "   a * v:\t" << timer.best() << "  " << double(REPEAT)/timer.best() << " * / sec " << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        v2 = m1.transpose() * v1;
      timer.stop();
      std::cout << "   a' * v:\t" << timer.best() << endl;
    }
    #endif

    // eigen sparse matrices
    {
      std::cout << "Eigen sparse\t" << sm1.nonZeros()/float(sm1.rows()*sm1.cols())*100 << "%\n";

      BENCH(asm("#myc"); v2 = sm1 * v1; asm("#myd");)
      std::cout << "   a * v:\t" << timer.best()/REPEAT << "  " << double(REPEAT)/timer.best(REAL_TIMER) << " * / sec " << endl;


      BENCH( { asm("#mya"); v2 = sm1.transpose() * v1; asm("#myb"); })

      std::cout << "   a' * v:\t" << timer.best()/REPEAT << endl;
    }

//     {
//       DynamicSparseMatrix<Scalar> m1(sm1);
//       std::cout << "Eigen dyn-sparse\t" << m1.nonZeros()/float(m1.rows()*m1.cols())*100 << "%\n";
//
//       BENCH(for (int k=0; k<REPEAT; ++k) v2 = m1 * v1;)
//       std::cout << "   a * v:\t" << timer.value() << endl;
//
//       BENCH(for (int k=0; k<REPEAT; ++k) v2 = m1.transpose() * v1;)
//       std::cout << "   a' * v:\t" << timer.value() << endl;
//     }

    // GMM++
    #ifndef NOGMM
    {
      std::cout << "GMM++ sparse\t" << density*100 << "%\n";
      //GmmDynSparse  gmmT3(rows,cols);
      GmmSparse m1(rows,cols);
      eiToGmm(sm1, m1);

      std::vector<Scalar> gmmV1(cols), gmmV2(cols);
      Map<Matrix<Scalar,Dynamic,1> >(&gmmV1[0], cols) = v1;
      Map<Matrix<Scalar,Dynamic,1> >(&gmmV2[0], cols) = v2;

      BENCH( asm("#myx"); gmm::mult(m1, gmmV1, gmmV2); asm("#myy"); )
      std::cout << "   a * v:\t" << timer.value() << endl;

      BENCH( gmm::mult(gmm::transposed(m1), gmmV1, gmmV2); )
      std::cout << "   a' * v:\t" << timer.value() << endl;
    }
    #endif
    
    #ifndef NOUBLAS
    {
      std::cout << "ublas sparse\t" << density*100 << "%\n";
      UBlasSparse m1(rows,cols);
      eiToUblas(sm1, m1);
      
      boost::numeric::ublas::vector<Scalar> uv1, uv2;
      eiToUblasVec(v1,uv1);
      eiToUblasVec(v2,uv2);

//       std::vector<Scalar> gmmV1(cols), gmmV2(cols);
//       Map<Matrix<Scalar,Dynamic,1> >(&gmmV1[0], cols) = v1;
//       Map<Matrix<Scalar,Dynamic,1> >(&gmmV2[0], cols) = v2;

      BENCH( uv2 = boost::numeric::ublas::prod(m1, uv1); )
      std::cout << "   a * v:\t" << timer.value() << endl;

//       BENCH( boost::ublas::prod(gmm::transposed(m1), gmmV1, gmmV2); )
//       std::cout << "   a' * v:\t" << timer.value() << endl;
    }
    #endif

    // MTL4
    #ifndef NOMTL
    {
      std::cout << "MTL4\t" << density*100 << "%\n";
      MtlSparse m1(rows,cols);
      eiToMtl(sm1, m1);
      mtl::dense_vector<Scalar> mtlV1(cols, 1.0);
      mtl::dense_vector<Scalar> mtlV2(cols, 1.0);

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        mtlV2 = m1 * mtlV1;
      timer.stop();
      std::cout << "   a * v:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        mtlV2 = trans(m1) * mtlV1;
      timer.stop();
      std::cout << "   a' * v:\t" << timer.value() << endl;
    }
    #endif

    std::cout << "\n\n";
  }

  return 0;
}

