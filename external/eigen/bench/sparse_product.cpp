
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.005 -DSIZE=10000 && ./a.out
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.05 -DSIZE=2000 && ./a.out
// -DNOGMM -DNOMTL -DCSPARSE
// -I /home/gael/Coding/LinearAlgebra/CSparse/Include/ /home/gael/Coding/LinearAlgebra/CSparse/Lib/libcsparse.a

#include <typeinfo>

#ifndef SIZE
#define SIZE 1000000
#endif

#ifndef NNZPERCOL
#define NNZPERCOL 6
#endif

#ifndef REPEAT
#define REPEAT 1
#endif

#include <algorithm>
#include "BenchTimer.h"
#include "BenchUtil.h"
#include "BenchSparseUtil.h"

#ifndef NBTRIES
#define NBTRIES 1
#endif

#define BENCH(X) \
  timer.reset(); \
  for (int _j=0; _j<NBTRIES; ++_j) { \
    timer.start(); \
    for (int _k=0; _k<REPEAT; ++_k) { \
        X  \
  } timer.stop(); }

// #ifdef MKL
//
// #include "mkl_types.h"
// #include "mkl_spblas.h"
//
// template<typename Lhs,typename Rhs,typename Res>
// void mkl_multiply(const Lhs& lhs, const Rhs& rhs, Res& res)
// {
//   char n = 'N';
//   float alpha = 1;
//   char matdescra[6];
//   matdescra[0] = 'G';
//   matdescra[1] = 0;
//   matdescra[2] = 0;
//   matdescra[3] = 'C';
//   mkl_scscmm(&n, lhs.rows(), rhs.cols(), lhs.cols(), &alpha, matdescra,
//              lhs._valuePtr(), lhs._innerIndexPtr(), lhs.outerIndexPtr(),
//              pntre, b, &ldb, &beta, c, &ldc);
// //   mkl_somatcopy('C', 'T', lhs.rows(), lhs.cols(), 1,
// //                 lhs._valuePtr(), lhs.rows(), DST, dst_stride);
// }
//
// #endif


#ifdef CSPARSE
cs* cs_sorted_multiply(const cs* a, const cs* b)
{
//   return cs_multiply(a,b);

  cs* A = cs_transpose(a, 1);
  cs* B = cs_transpose(b, 1);
  cs* D = cs_multiply(B,A);   /* D = B'*A' */
  cs_spfree (A) ;
  cs_spfree (B) ;
  cs_dropzeros (D) ;      /* drop zeros from D */
  cs* C = cs_transpose (D, 1) ;   /* C = D', so that C is sorted */
  cs_spfree (D) ;
  return C;

//   cs* A = cs_transpose(a, 1);
//   cs* C = cs_transpose(A, 1);
//   return C;
}

cs* cs_sorted_multiply2(const cs* a, const cs* b)
{
  cs* D = cs_multiply(a,b);
  cs* E = cs_transpose(D,1);
  cs_spfree(D);
  cs* C = cs_transpose(E,1);
  cs_spfree(E);
  return C;
}
#endif

void bench_sort();

int main(int argc, char *argv[])
{
//   bench_sort();

  int rows = SIZE;
  int cols = SIZE;
  float density = DENSITY;

  EigenSparseMatrix sm1(rows,cols), sm2(rows,cols), sm3(rows,cols), sm4(rows,cols);

  BenchTimer timer;
  for (int nnzPerCol = NNZPERCOL; nnzPerCol>1; nnzPerCol/=1.1)
  {
    sm1.setZero();
    sm2.setZero();
    fillMatrix2(nnzPerCol, rows, cols, sm1);
    fillMatrix2(nnzPerCol, rows, cols, sm2);
//     std::cerr << "filling OK\n";

    // dense matrices
    #ifdef DENSEMATRIX
    {
      std::cout << "Eigen Dense\t" << nnzPerCol << "%\n";
      DenseMatrix m1(rows,cols), m2(rows,cols), m3(rows,cols);
      eiToDense(sm1, m1);
      eiToDense(sm2, m2);

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1 * m2;
      timer.stop();
      std::cout << "   a * b:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1.transpose() * m2;
      timer.stop();
      std::cout << "   a' * b:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1.transpose() * m2.transpose();
      timer.stop();
      std::cout << "   a' * b':\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1 * m2.transpose();
      timer.stop();
      std::cout << "   a * b':\t" << timer.value() << endl;
    }
    #endif

    // eigen sparse matrices
    {
      std::cout << "Eigen sparse\t" << sm1.nonZeros()/(float(sm1.rows())*float(sm1.cols()))*100 << "% * "
                << sm2.nonZeros()/(float(sm2.rows())*float(sm2.cols()))*100 << "%\n";

      BENCH(sm3 = sm1 * sm2; )
      std::cout << "   a * b:\t" << timer.value() << endl;

//       BENCH(sm3 = sm1.transpose() * sm2; )
//       std::cout << "   a' * b:\t" << timer.value() << endl;
// //
//       BENCH(sm3 = sm1.transpose() * sm2.transpose(); )
//       std::cout << "   a' * b':\t" << timer.value() << endl;
// //
//       BENCH(sm3 = sm1 * sm2.transpose(); )
//       std::cout << "   a * b' :\t" << timer.value() << endl;


//       std::cout << "\n";
//
//       BENCH( sm3._experimentalNewProduct(sm1, sm2); )
//       std::cout << "   a * b:\t" << timer.value() << endl;
//
//       BENCH(sm3._experimentalNewProduct(sm1.transpose(),sm2); )
//       std::cout << "   a' * b:\t" << timer.value() << endl;
// //
//       BENCH(sm3._experimentalNewProduct(sm1.transpose(),sm2.transpose()); )
//       std::cout << "   a' * b':\t" << timer.value() << endl;
// //
//       BENCH(sm3._experimentalNewProduct(sm1, sm2.transpose());)
//       std::cout << "   a * b' :\t" << timer.value() << endl;
    }

    // eigen dyn-sparse matrices
    /*{
      DynamicSparseMatrix<Scalar> m1(sm1), m2(sm2), m3(sm3);
      std::cout << "Eigen dyn-sparse\t" << m1.nonZeros()/(float(m1.rows())*float(m1.cols()))*100 << "% * "
                << m2.nonZeros()/(float(m2.rows())*float(m2.cols()))*100 << "%\n";

//       timer.reset();
//       timer.start();
      BENCH(for (int k=0; k<REPEAT; ++k) m3 = m1 * m2;)
//       timer.stop();
      std::cout << "   a * b:\t" << timer.value() << endl;
//       std::cout << sm3 << "\n";

      timer.reset();
      timer.start();
//       std::cerr << "transpose...\n";
//       EigenSparseMatrix sm4 = sm1.transpose();
//       std::cout << sm4.nonZeros() << " == " << sm1.nonZeros() << "\n";
//       exit(1);
//       std::cerr << "transpose OK\n";
//       std::cout << sm1 << "\n\n" << sm1.transpose() << "\n\n" << sm4.transpose() << "\n\n";
      BENCH(for (int k=0; k<REPEAT; ++k) m3 = m1.transpose() * m2;)
//       timer.stop();
      std::cout << "   a' * b:\t" << timer.value() << endl;

//       timer.reset();
//       timer.start();
      BENCH( for (int k=0; k<REPEAT; ++k) m3 = m1.transpose() * m2.transpose(); )
//       timer.stop();
      std::cout << "   a' * b':\t" << timer.value() << endl;

//       timer.reset();
//       timer.start();
      BENCH( for (int k=0; k<REPEAT; ++k) m3 = m1 * m2.transpose(); )
//       timer.stop();
      std::cout << "   a * b' :\t" << timer.value() << endl;
    }*/

    // CSparse
    #ifdef CSPARSE
    {
      std::cout << "CSparse \t" << nnzPerCol << "%\n";
      cs *m1, *m2, *m3;
      eiToCSparse(sm1, m1);
      eiToCSparse(sm2, m2);

      BENCH(
      {
        m3 = cs_sorted_multiply(m1, m2);
        if (!m3)
        {
          std::cerr << "cs_multiply failed\n";
        }
//         cs_print(m3, 0);
        cs_spfree(m3);
      }
      );
//       timer.stop();
      std::cout << "   a * b:\t" << timer.value() << endl;

//       BENCH( { m3 = cs_sorted_multiply2(m1, m2); cs_spfree(m3); } );
//       std::cout << "   a * b:\t" << timer.value() << endl;
    }
    #endif

    #ifndef NOUBLAS
    {
      std::cout << "ublas\t" << nnzPerCol << "%\n";
      UBlasSparse m1(rows,cols), m2(rows,cols), m3(rows,cols);
      eiToUblas(sm1, m1);
      eiToUblas(sm2, m2);

      BENCH(boost::numeric::ublas::prod(m1, m2, m3););
      std::cout << "   a * b:\t" << timer.value() << endl;
    }
    #endif

    // GMM++
    #ifndef NOGMM
    {
      std::cout << "GMM++ sparse\t" << nnzPerCol << "%\n";
      GmmDynSparse  gmmT3(rows,cols);
      GmmSparse m1(rows,cols), m2(rows,cols), m3(rows,cols);
      eiToGmm(sm1, m1);
      eiToGmm(sm2, m2);

      BENCH(gmm::mult(m1, m2, gmmT3););
      std::cout << "   a * b:\t" << timer.value() << endl;

//       BENCH(gmm::mult(gmm::transposed(m1), m2, gmmT3););
//       std::cout << "   a' * b:\t" << timer.value() << endl;
//
//       if (rows<500)
//       {
//         BENCH(gmm::mult(gmm::transposed(m1), gmm::transposed(m2), gmmT3););
//         std::cout << "   a' * b':\t" << timer.value() << endl;
//
//         BENCH(gmm::mult(m1, gmm::transposed(m2), gmmT3););
//         std::cout << "   a * b':\t" << timer.value() << endl;
//       }
//       else
//       {
//         std::cout << "   a' * b':\t" << "forever" << endl;
//         std::cout << "   a * b':\t" << "forever" << endl;
//       }
    }
    #endif

    // MTL4
    #ifndef NOMTL
    {
      std::cout << "MTL4\t" << nnzPerCol << "%\n";
      MtlSparse m1(rows,cols), m2(rows,cols), m3(rows,cols);
      eiToMtl(sm1, m1);
      eiToMtl(sm2, m2);

      BENCH(m3 = m1 * m2;);
      std::cout << "   a * b:\t" << timer.value() << endl;

//       BENCH(m3 = trans(m1) * m2;);
//       std::cout << "   a' * b:\t" << timer.value() << endl;
//
//       BENCH(m3 = trans(m1) * trans(m2););
//       std::cout << "  a' * b':\t" << timer.value() << endl;
//
//       BENCH(m3 = m1 * trans(m2););
//       std::cout << "   a * b' :\t" << timer.value() << endl;
    }
    #endif

    std::cout << "\n\n";
  }

  return 0;
}



