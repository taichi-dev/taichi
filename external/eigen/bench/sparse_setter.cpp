
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.005 -DSIZE=10000 && ./a.out
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.05 -DSIZE=2000 && ./a.out
// -DNOGMM -DNOMTL -DCSPARSE
// -I /home/gael/Coding/LinearAlgebra/CSparse/Include/ /home/gael/Coding/LinearAlgebra/CSparse/Lib/libcsparse.a
#ifndef SIZE
#define SIZE 100000
#endif

#ifndef NBPERROW
#define NBPERROW 24
#endif

#ifndef REPEAT
#define REPEAT 2
#endif

#ifndef NBTRIES
#define NBTRIES 2
#endif

#ifndef KK
#define KK 10
#endif

#ifndef NOGOOGLE
#define EIGEN_GOOGLEHASH_SUPPORT
#include <google/sparse_hash_map>
#endif

#include "BenchSparseUtil.h"

#define CHECK_MEM
// #define CHECK_MEM  std/**/::cout << "check mem\n"; getchar();

#define BENCH(X) \
  timer.reset(); \
  for (int _j=0; _j<NBTRIES; ++_j) { \
    timer.start(); \
    for (int _k=0; _k<REPEAT; ++_k) { \
        X  \
  } timer.stop(); }

typedef std::vector<Vector2i> Coordinates;
typedef std::vector<float> Values;

EIGEN_DONT_INLINE Scalar* setinnerrand_eigen(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_eigen_dynamic(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_eigen_compact(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_eigen_sumeq(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_eigen_gnu_hash(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_eigen_google_dense(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_eigen_google_sparse(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_scipy(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_ublas_mapped(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_ublas_coord(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_ublas_compressed(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_ublas_genvec(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_mtl(const Coordinates& coords, const Values& vals);

int main(int argc, char *argv[])
{
  int rows = SIZE;
  int cols = SIZE;
  bool fullyrand = true;

  BenchTimer timer;
  Coordinates coords;
  Values values;
  if(fullyrand)
  {
    Coordinates pool;
    pool.reserve(cols*NBPERROW);
    std::cerr << "fill pool" << "\n";
    for (int i=0; i<cols*NBPERROW; )
    {
//       DynamicSparseMatrix<int> stencil(SIZE,SIZE);
      Vector2i ij(internal::random<int>(0,rows-1),internal::random<int>(0,cols-1));
//       if(stencil.coeffRef(ij.x(), ij.y())==0)
      {
//         stencil.coeffRef(ij.x(), ij.y()) = 1;
        pool.push_back(ij);

      }
      ++i;
    }
    std::cerr << "pool ok" << "\n";
    int n = cols*NBPERROW*KK;
    coords.reserve(n);
    values.reserve(n);
    for (int i=0; i<n; ++i)
    {
      int i = internal::random<int>(0,pool.size());
      coords.push_back(pool[i]);
      values.push_back(internal::random<Scalar>());
    }
  }
  else
  {
    for (int j=0; j<cols; ++j)
    for (int i=0; i<NBPERROW; ++i)
    {
      coords.push_back(Vector2i(internal::random<int>(0,rows-1),j));
      values.push_back(internal::random<Scalar>());
    }
  }
  std::cout << "nnz = " << coords.size()  << "\n";
  CHECK_MEM

    // dense matrices
    #ifdef DENSEMATRIX
    {
      BENCH(setrand_eigen_dense(coords,values);)
      std::cout << "Eigen Dense\t" << timer.value() << "\n";
    }
    #endif

    // eigen sparse matrices
//     if (!fullyrand)
//     {
//       BENCH(setinnerrand_eigen(coords,values);)
//       std::cout << "Eigen fillrand\t" << timer.value() << "\n";
//     }
    {
      BENCH(setrand_eigen_dynamic(coords,values);)
      std::cout << "Eigen dynamic\t" << timer.value() << "\n";
    }
//     {
//       BENCH(setrand_eigen_compact(coords,values);)
//       std::cout << "Eigen compact\t" << timer.value() << "\n";
//     }
    {
      BENCH(setrand_eigen_sumeq(coords,values);)
      std::cout << "Eigen sumeq\t" << timer.value() << "\n";
    }
    {
//       BENCH(setrand_eigen_gnu_hash(coords,values);)
//       std::cout << "Eigen std::map\t" << timer.value() << "\n";
    }
    {
      BENCH(setrand_scipy(coords,values);)
      std::cout << "scipy\t" << timer.value() << "\n";
    }
    #ifndef NOGOOGLE
    {
      BENCH(setrand_eigen_google_dense(coords,values);)
      std::cout << "Eigen google dense\t" << timer.value() << "\n";
    }
    {
      BENCH(setrand_eigen_google_sparse(coords,values);)
      std::cout << "Eigen google sparse\t" << timer.value() << "\n";
    }
    #endif

    #ifndef NOUBLAS
    {
//       BENCH(setrand_ublas_mapped(coords,values);)
//       std::cout << "ublas mapped\t" << timer.value() << "\n";
    }
    {
      BENCH(setrand_ublas_genvec(coords,values);)
      std::cout << "ublas vecofvec\t" << timer.value() << "\n";
    }
    /*{
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_ublas_compressed(coords,values);
      timer.stop();
      std::cout << "ublas comp\t" << timer.value() << "\n";
    }
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_ublas_coord(coords,values);
      timer.stop();
      std::cout << "ublas coord\t" << timer.value() << "\n";
    }*/
    #endif


    // MTL4
    #ifndef NOMTL
    {
      BENCH(setrand_mtl(coords,values));
      std::cout << "MTL\t" << timer.value() << "\n";
    }
    #endif

  return 0;
}

EIGEN_DONT_INLINE Scalar* setinnerrand_eigen(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  SparseMatrix<Scalar> mat(SIZE,SIZE);
  //mat.startFill(2000000/*coords.size()*/);
  for (int i=0; i<coords.size(); ++i)
  {
    mat.insert(coords[i].x(), coords[i].y()) = vals[i];
  }
  mat.finalize();
  CHECK_MEM;
  return 0;
}

EIGEN_DONT_INLINE Scalar* setrand_eigen_dynamic(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  DynamicSparseMatrix<Scalar> mat(SIZE,SIZE);
  mat.reserve(coords.size()/10);
  for (int i=0; i<coords.size(); ++i)
  {
    mat.coeffRef(coords[i].x(), coords[i].y()) += vals[i];
  }
  mat.finalize();
  CHECK_MEM;
  return &mat.coeffRef(coords[0].x(), coords[0].y());
}

EIGEN_DONT_INLINE Scalar* setrand_eigen_sumeq(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  int n = coords.size()/KK;
  DynamicSparseMatrix<Scalar> mat(SIZE,SIZE);
  for (int j=0; j<KK; ++j)
  {
    DynamicSparseMatrix<Scalar> aux(SIZE,SIZE);
    mat.reserve(n);
    for (int i=j*n; i<(j+1)*n; ++i)
    {
      aux.insert(coords[i].x(), coords[i].y()) += vals[i];
    }
    aux.finalize();
    mat += aux;
  }
  return &mat.coeffRef(coords[0].x(), coords[0].y());
}

EIGEN_DONT_INLINE Scalar* setrand_eigen_compact(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  DynamicSparseMatrix<Scalar> setter(SIZE,SIZE);
  setter.reserve(coords.size()/10);
  for (int i=0; i<coords.size(); ++i)
  {
    setter.coeffRef(coords[i].x(), coords[i].y()) += vals[i];
  }
  SparseMatrix<Scalar> mat = setter;
  CHECK_MEM;
  return &mat.coeffRef(coords[0].x(), coords[0].y());
}

EIGEN_DONT_INLINE Scalar* setrand_eigen_gnu_hash(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  SparseMatrix<Scalar> mat(SIZE,SIZE);
  {
    RandomSetter<SparseMatrix<Scalar>, StdMapTraits > setter(mat);
    for (int i=0; i<coords.size(); ++i)
    {
      setter(coords[i].x(), coords[i].y()) += vals[i];
    }
    CHECK_MEM;
  }
  return &mat.coeffRef(coords[0].x(), coords[0].y());
}

#ifndef NOGOOGLE
EIGEN_DONT_INLINE Scalar* setrand_eigen_google_dense(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  SparseMatrix<Scalar> mat(SIZE,SIZE);
  {
    RandomSetter<SparseMatrix<Scalar>, GoogleDenseHashMapTraits> setter(mat);
    for (int i=0; i<coords.size(); ++i)
      setter(coords[i].x(), coords[i].y()) += vals[i];
    CHECK_MEM;
  }
  return &mat.coeffRef(coords[0].x(), coords[0].y());
}

EIGEN_DONT_INLINE Scalar* setrand_eigen_google_sparse(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  SparseMatrix<Scalar> mat(SIZE,SIZE);
  {
    RandomSetter<SparseMatrix<Scalar>, GoogleSparseHashMapTraits> setter(mat);
    for (int i=0; i<coords.size(); ++i)
      setter(coords[i].x(), coords[i].y()) += vals[i];
    CHECK_MEM;
  }
  return &mat.coeffRef(coords[0].x(), coords[0].y());
}
#endif


template <class T>
void coo_tocsr(const int n_row,
               const int n_col,
               const int nnz,
               const Coordinates Aij,
               const Values Ax,
                     int Bp[],
                     int Bj[],
                     T Bx[])
{
    //compute number of non-zero entries per row of A coo_tocsr
    std::fill(Bp, Bp + n_row, 0);

    for (int n = 0; n < nnz; n++){
        Bp[Aij[n].x()]++;
    }

    //cumsum the nnz per row to get Bp[]
    for(int i = 0, cumsum = 0; i < n_row; i++){
        int temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz;

    //write Aj,Ax into Bj,Bx
    for(int n = 0; n < nnz; n++){
        int row  = Aij[n].x();
        int dest = Bp[row];

        Bj[dest] = Aij[n].y();
        Bx[dest] = Ax[n];

        Bp[row]++;
    }

    for(int i = 0, last = 0; i <= n_row; i++){
        int temp = Bp[i];
        Bp[i]  = last;
        last   = temp;
    }

    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

template< class T1, class T2 >
bool kv_pair_less(const std::pair<T1,T2>& x, const std::pair<T1,T2>& y){
    return x.first < y.first;
}


template<class I, class T>
void csr_sort_indices(const I n_row,
                      const I Ap[],
                            I Aj[],
                            T Ax[])
{
    std::vector< std::pair<I,T> > temp;

    for(I i = 0; i < n_row; i++){
        I row_start = Ap[i];
        I row_end   = Ap[i+1];

        temp.clear();

        for(I jj = row_start; jj < row_end; jj++){
            temp.push_back(std::make_pair(Aj[jj],Ax[jj]));
        }

        std::sort(temp.begin(),temp.end(),kv_pair_less<I,T>);

        for(I jj = row_start, n = 0; jj < row_end; jj++, n++){
            Aj[jj] = temp[n].first;
            Ax[jj] = temp[n].second;
        }
    }
}

template <class I, class T>
void csr_sum_duplicates(const I n_row,
                        const I n_col,
                              I Ap[],
                              I Aj[],
                              T Ax[])
{
    I nnz = 0;
    I row_end = 0;
    for(I i = 0; i < n_row; i++){
        I jj = row_end;
        row_end = Ap[i+1];
        while( jj < row_end ){
            I j = Aj[jj];
            T x = Ax[jj];
            jj++;
            while( jj < row_end && Aj[jj] == j ){
                x += Ax[jj];
                jj++;
            }
            Aj[nnz] = j;
            Ax[nnz] = x;
            nnz++;
        }
        Ap[i+1] = nnz;
    }
}

EIGEN_DONT_INLINE Scalar* setrand_scipy(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  SparseMatrix<Scalar> mat(SIZE,SIZE);
  mat.resizeNonZeros(coords.size());
//   std::cerr << "setrand_scipy...\n";
  coo_tocsr<Scalar>(SIZE,SIZE, coords.size(), coords, vals, mat._outerIndexPtr(), mat._innerIndexPtr(), mat._valuePtr());
//   std::cerr << "coo_tocsr ok\n";

  csr_sort_indices(SIZE, mat._outerIndexPtr(), mat._innerIndexPtr(), mat._valuePtr());

  csr_sum_duplicates(SIZE, SIZE, mat._outerIndexPtr(), mat._innerIndexPtr(), mat._valuePtr());

  mat.resizeNonZeros(mat._outerIndexPtr()[SIZE]);

  return &mat.coeffRef(coords[0].x(), coords[0].y());
}


#ifndef NOUBLAS
EIGEN_DONT_INLINE Scalar* setrand_ublas_mapped(const Coordinates& coords, const Values& vals)
{
  using namespace boost;
  using namespace boost::numeric;
  using namespace boost::numeric::ublas;
  mapped_matrix<Scalar> aux(SIZE,SIZE);
  for (int i=0; i<coords.size(); ++i)
  {
    aux(coords[i].x(), coords[i].y()) += vals[i];
  }
  CHECK_MEM;
  compressed_matrix<Scalar> mat(aux);
  return 0;// &mat(coords[0].x(), coords[0].y());
}
/*EIGEN_DONT_INLINE Scalar* setrand_ublas_coord(const Coordinates& coords, const Values& vals)
{
  using namespace boost;
  using namespace boost::numeric;
  using namespace boost::numeric::ublas;
  coordinate_matrix<Scalar> aux(SIZE,SIZE);
  for (int i=0; i<coords.size(); ++i)
  {
    aux(coords[i].x(), coords[i].y()) = vals[i];
  }
  compressed_matrix<Scalar> mat(aux);
  return 0;//&mat(coords[0].x(), coords[0].y());
}
EIGEN_DONT_INLINE Scalar* setrand_ublas_compressed(const Coordinates& coords, const Values& vals)
{
  using namespace boost;
  using namespace boost::numeric;
  using namespace boost::numeric::ublas;
  compressed_matrix<Scalar> mat(SIZE,SIZE);
  for (int i=0; i<coords.size(); ++i)
  {
    mat(coords[i].x(), coords[i].y()) = vals[i];
  }
  return 0;//&mat(coords[0].x(), coords[0].y());
}*/
EIGEN_DONT_INLINE Scalar* setrand_ublas_genvec(const Coordinates& coords, const Values& vals)
{
  using namespace boost;
  using namespace boost::numeric;
  using namespace boost::numeric::ublas;

//   ublas::vector<coordinate_vector<Scalar> > foo;
  generalized_vector_of_vector<Scalar, row_major, ublas::vector<coordinate_vector<Scalar> > > aux(SIZE,SIZE);
  for (int i=0; i<coords.size(); ++i)
  {
    aux(coords[i].x(), coords[i].y()) += vals[i];
  }
  CHECK_MEM;
  compressed_matrix<Scalar,row_major> mat(aux);
  return 0;//&mat(coords[0].x(), coords[0].y());
}
#endif

#ifndef NOMTL
EIGEN_DONT_INLINE void setrand_mtl(const Coordinates& coords, const Values& vals);
#endif

