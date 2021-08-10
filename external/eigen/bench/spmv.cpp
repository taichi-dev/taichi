
//g++-4.4 -DNOMTL  -Wl,-rpath /usr/local/lib/oski -L /usr/local/lib/oski/ -l oski -l oski_util -l oski_util_Tid  -DOSKI -I ~/Coding/LinearAlgebra/mtl4/  spmv.cpp  -I .. -O2 -DNDEBUG -lrt  -lm -l oski_mat_CSC_Tid  -loskilt && ./a.out r200000 c200000 n100 t1 p1

#define SCALAR double

#include <iostream>
#include <algorithm>
#include "BenchTimer.h"
#include "BenchSparseUtil.h"

#define SPMV_BENCH(CODE) BENCH(t,tries,repeats,CODE);

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

int main(int argc, char *argv[])
{
  int size = 10000;
  int rows = size;
  int cols = size;
  int nnzPerCol = 40;
  int tries = 2;
  int repeats = 2;

  bool need_help = false;
  for(int i = 1; i < argc; i++)
  {
    if(argv[i][0] == 'r')
    {
      rows = atoi(argv[i]+1);
    }
    else if(argv[i][0] == 'c')
    {
      cols = atoi(argv[i]+1);
    }
    else if(argv[i][0] == 'n')
    {
      nnzPerCol = atoi(argv[i]+1);
    }
    else if(argv[i][0] == 't')
    {
      tries = atoi(argv[i]+1);
    }
    else if(argv[i][0] == 'p')
    {
      repeats = atoi(argv[i]+1);
    }
    else
    {
      need_help = true;
    }
  }
  if(need_help)
  {
    std::cout << argv[0] << " r<nb rows> c<nb columns> n<non zeros per column> t<nb tries> p<nb repeats>\n";
    return 1;
  }

  std::cout << "SpMV " << rows << " x " << cols << " with " << nnzPerCol << " non zeros per column. (" << repeats << " repeats, and " << tries << " tries)\n\n";

  EigenSparseMatrix sm(rows,cols);
  DenseVector dv(cols), res(rows);
  dv.setRandom();

  BenchTimer t;
  while (nnzPerCol>=4)
  {
    std::cout << "nnz: " << nnzPerCol << "\n";
    sm.setZero();
    fillMatrix2(nnzPerCol, rows, cols, sm);

    // dense matrices
    #ifdef DENSEMATRIX
    {
      DenseMatrix dm(rows,cols), (rows,cols);
      eiToDense(sm, dm);

      SPMV_BENCH(res = dm * sm);
      std::cout << "Dense       " << t.value()/repeats << "\t";

      SPMV_BENCH(res = dm.transpose() * sm);
      std::cout << t.value()/repeats << endl;
    }
    #endif

    // eigen sparse matrices
    {
      SPMV_BENCH(res.noalias() += sm * dv; )
      std::cout << "Eigen       " << t.value()/repeats << "\t";

      SPMV_BENCH(res.noalias() += sm.transpose() * dv; )
      std::cout << t.value()/repeats << endl;
    }

    // CSparse
    #ifdef CSPARSE
    {
      std::cout << "CSparse \n";
      cs *csm;
      eiToCSparse(sm, csm);

//       BENCH();
//       timer.stop();
//       std::cout << "   a * b:\t" << timer.value() << endl;

//       BENCH( { m3 = cs_sorted_multiply2(m1, m2); cs_spfree(m3); } );
//       std::cout << "   a * b:\t" << timer.value() << endl;
    }
    #endif

    #ifdef OSKI
    {
      oski_matrix_t om;
      oski_vecview_t ov, ores;
      oski_Init();
      om = oski_CreateMatCSC(sm._outerIndexPtr(), sm._innerIndexPtr(), sm._valuePtr(), rows, cols,
                             SHARE_INPUTMAT, 1, INDEX_ZERO_BASED);
      ov = oski_CreateVecView(dv.data(), cols, STRIDE_UNIT);
      ores = oski_CreateVecView(res.data(), rows, STRIDE_UNIT);

      SPMV_BENCH( oski_MatMult(om, OP_NORMAL, 1, ov, 0, ores) );
      std::cout << "OSKI        " << t.value()/repeats << "\t";

      SPMV_BENCH( oski_MatMult(om, OP_TRANS, 1, ov, 0, ores) );
      std::cout << t.value()/repeats << "\n";

      // tune
      t.reset();
      t.start();
      oski_SetHintMatMult(om, OP_NORMAL, 1.0, SYMBOLIC_VEC, 0.0, SYMBOLIC_VEC, ALWAYS_TUNE_AGGRESSIVELY);
      oski_TuneMat(om);
      t.stop();
      double tuning = t.value();

      SPMV_BENCH( oski_MatMult(om, OP_NORMAL, 1, ov, 0, ores) );
      std::cout << "OSKI tuned  " << t.value()/repeats << "\t";

      SPMV_BENCH( oski_MatMult(om, OP_TRANS, 1, ov, 0, ores) );
      std::cout << t.value()/repeats << "\t(" << tuning <<  ")\n";


      oski_DestroyMat(om);
      oski_DestroyVecView(ov);
      oski_DestroyVecView(ores);
      oski_Close();
    }
    #endif

    #ifndef NOUBLAS
    {
      using namespace boost::numeric;
      UblasMatrix um(rows,cols);
      eiToUblas(sm, um);

      boost::numeric::ublas::vector<Scalar> uv(cols), ures(rows);
      Map<Matrix<Scalar,Dynamic,1> >(&uv[0], cols) = dv;
      Map<Matrix<Scalar,Dynamic,1> >(&ures[0], rows) = res;

      SPMV_BENCH(ublas::axpy_prod(um, uv, ures, true));
      std::cout << "ublas       " << t.value()/repeats << "\t";

      SPMV_BENCH(ublas::axpy_prod(boost::numeric::ublas::trans(um), uv, ures, true));
      std::cout << t.value()/repeats << endl;
    }
    #endif

    // GMM++
    #ifndef NOGMM
    {
      GmmSparse gm(rows,cols);
      eiToGmm(sm, gm);

      std::vector<Scalar> gv(cols), gres(rows);
      Map<Matrix<Scalar,Dynamic,1> >(&gv[0], cols) = dv;
      Map<Matrix<Scalar,Dynamic,1> >(&gres[0], rows) = res;

      SPMV_BENCH(gmm::mult(gm, gv, gres));
      std::cout << "GMM++       " << t.value()/repeats << "\t";

      SPMV_BENCH(gmm::mult(gmm::transposed(gm), gv, gres));
      std::cout << t.value()/repeats << endl;
    }
    #endif

    // MTL4
    #ifndef NOMTL
    {
      MtlSparse mm(rows,cols);
      eiToMtl(sm, mm);
      mtl::dense_vector<Scalar> mv(cols, 1.0);
      mtl::dense_vector<Scalar> mres(rows, 1.0);

      SPMV_BENCH(mres = mm * mv);
      std::cout << "MTL4        " << t.value()/repeats << "\t";

      SPMV_BENCH(mres = trans(mm) * mv);
      std::cout << t.value()/repeats << endl;
    }
    #endif

    std::cout << "\n";

    if(nnzPerCol==1)
      break;
    nnzPerCol -= nnzPerCol/2;
  }

  return 0;
}



