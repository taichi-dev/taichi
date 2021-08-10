
#define BLAS_FUNC(NAME) CAT(CAT(SCALAR_PREFIX,NAME),_)

template<> class blas_interface<SCALAR> : public c_interface_base<SCALAR>
{

public :
  
  static SCALAR fone;
  static SCALAR fzero;

  static inline std::string name()
  {
    return MAKE_STRING(CBLASNAME);
  }

  static inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    BLAS_FUNC(gemv)(&notrans,&N,&N,&fone,A,&N,B,&intone,&fzero,X,&intone);
  }

  static inline void symv(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    BLAS_FUNC(symv)(&lower, &N,&fone,A,&N,B,&intone,&fzero,X,&intone);
  }

  static inline void syr2(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    BLAS_FUNC(syr2)(&lower,&N,&fone,B,&intone,X,&intone,A,&N);
  }

  static inline void ger(gene_matrix & A, gene_vector & X, gene_vector & Y, int N){
    BLAS_FUNC(ger)(&N,&N,&fone,X,&intone,Y,&intone,A,&N);
  }

  static inline void rot(gene_vector & A,  gene_vector & B, SCALAR c, SCALAR s, int N){
    BLAS_FUNC(rot)(&N,A,&intone,B,&intone,&c,&s);
  }

  static inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    BLAS_FUNC(gemv)(&trans,&N,&N,&fone,A,&N,B,&intone,&fzero,X,&intone);
  }

  static inline void matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N){
    BLAS_FUNC(gemm)(&notrans,&notrans,&N,&N,&N,&fone,A,&N,B,&N,&fzero,X,&N);
  }

  static inline void transposed_matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N){
    BLAS_FUNC(gemm)(&notrans,&notrans,&N,&N,&N,&fone,A,&N,B,&N,&fzero,X,&N);
  }

//   static inline void ata_product(gene_matrix & A, gene_matrix & X, int N){
//     ssyrk_(&lower,&trans,&N,&N,&fone,A,&N,&fzero,X,&N);
//   }

  static inline void aat_product(gene_matrix & A, gene_matrix & X, int N){
    BLAS_FUNC(syrk)(&lower,&notrans,&N,&N,&fone,A,&N,&fzero,X,&N);
  }

  static inline void axpy(SCALAR coef, const gene_vector & X, gene_vector & Y, int N){
    BLAS_FUNC(axpy)(&N,&coef,X,&intone,Y,&intone);
  }

  static inline void axpby(SCALAR a, const gene_vector & X, SCALAR b, gene_vector & Y, int N){
    BLAS_FUNC(scal)(&N,&b,Y,&intone);
    BLAS_FUNC(axpy)(&N,&a,X,&intone,Y,&intone);
  }

  static inline void cholesky(const gene_matrix & X, gene_matrix & C, int N){
    int N2 = N*N;
    BLAS_FUNC(copy)(&N2, X, &intone, C, &intone);
    char uplo = 'L';
    int info = 0;
    BLAS_FUNC(potrf)(&uplo, &N, C, &N, &info);
    if(info!=0) std::cerr << "potrf_ error " << info << "\n";
  }

  static inline void partial_lu_decomp(const gene_matrix & X, gene_matrix & C, int N){
    int N2 = N*N;
    BLAS_FUNC(copy)(&N2, X, &intone, C, &intone);
    int info = 0;
    int * ipiv = (int*)alloca(sizeof(int)*N);
    BLAS_FUNC(getrf)(&N, &N, C, &N, ipiv, &info);
    if(info!=0) std::cerr << "getrf_ error " << info << "\n";
  }
  
  static inline void trisolve_lower(const gene_matrix & L, const gene_vector& B, gene_vector & X, int N){
    BLAS_FUNC(copy)(&N, B, &intone, X, &intone);
    BLAS_FUNC(trsv)(&lower, &notrans, &nonunit, &N, L, &N, X, &intone);
  }

  static inline void trisolve_lower_matrix(const gene_matrix & L, const gene_matrix& B, gene_matrix & X, int N){
    BLAS_FUNC(copy)(&N, B, &intone, X, &intone);
    BLAS_FUNC(trsm)(&right, &lower, &notrans, &nonunit, &N, &N, &fone, L, &N, X, &N);
  }

  static inline void trmm(gene_matrix & A, gene_matrix & B, gene_matrix & /*X*/, int N){
    BLAS_FUNC(trmm)(&left, &lower, &notrans,&nonunit, &N,&N,&fone,A,&N,B,&N);
  }

  #ifdef HAS_LAPACK

  static inline void lu_decomp(const gene_matrix & X, gene_matrix & C, int N){
    int N2 = N*N;
    BLAS_FUNC(copy)(&N2, X, &intone, C, &intone);
    int info = 0;
    int * ipiv = (int*)alloca(sizeof(int)*N);
    int * jpiv = (int*)alloca(sizeof(int)*N);
    BLAS_FUNC(getc2)(&N, C, &N, ipiv, jpiv, &info);
  }



  static inline void hessenberg(const gene_matrix & X, gene_matrix & C, int N){
    {
      int N2 = N*N;
      int inc = 1;
      BLAS_FUNC(copy)(&N2, X, &inc, C, &inc);
    }
    int info = 0;
    int ilo = 1;
    int ihi = N;
    int bsize = 64;
    int worksize = N*bsize;
    SCALAR* d = new SCALAR[N+worksize];
    BLAS_FUNC(gehrd)(&N, &ilo, &ihi, C, &N, d, d+N, &worksize, &info);
    delete[] d;
  }

  static inline void tridiagonalization(const gene_matrix & X, gene_matrix & C, int N){
    {
      int N2 = N*N;
      int inc = 1;
      BLAS_FUNC(copy)(&N2, X, &inc, C, &inc);
    }
    char uplo = 'U';
    int info = 0;
    int bsize = 64;
    int worksize = N*bsize;
    SCALAR* d = new SCALAR[3*N+worksize];
    BLAS_FUNC(sytrd)(&uplo, &N, C, &N, d, d+N, d+2*N, d+3*N, &worksize, &info);
    delete[] d;
  }
  
  #endif // HAS_LAPACK

};

SCALAR blas_interface<SCALAR>::fone = SCALAR(1);
SCALAR blas_interface<SCALAR>::fzero = SCALAR(0);
