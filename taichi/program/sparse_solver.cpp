#include "taichi/ir/type_utils.h"

#include "sparse_solver.h"

#include <unordered_map>

namespace taichi::lang {
#define EIGEN_LLT_SOLVER_INSTANTIATION(dt, type, order)              \
  template class EigenSparseSolver<                                  \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                              Eigen::order##Ordering<int>>,          \
      Eigen::SparseMatrix<dt>>;
#define EIGEN_LU_SOLVER_INSTANTIATION(dt, type, order)  \
  template class EigenSparseSolver<                     \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,      \
                          Eigen::order##Ordering<int>>, \
      Eigen::SparseMatrix<dt>>;
// Explicit instantiation of EigenSparseSolver
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LLT, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LDLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LDLT, COLAMD);
EIGEN_LU_SOLVER_INSTANTIATION(float32, LU, AMD);
EIGEN_LU_SOLVER_INSTANTIATION(float32, LU, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LLT, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LDLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LDLT, COLAMD);
EIGEN_LU_SOLVER_INSTANTIATION(float64, LU, AMD);
EIGEN_LU_SOLVER_INSTANTIATION(float64, LU, COLAMD);
}  // namespace taichi::lang

// Explicit instantiation of the template class EigenSparseSolver::solve
#define EIGEN_LLT_SOLVE_INSTANTIATION(dt, type, order, df)               \
  using T##dt = Eigen::VectorX##df;                                      \
  using S##dt##type##order =                                             \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower,     \
                              Eigen::order##Ordering<int>>;              \
  template T##dt                                                         \
  EigenSparseSolver<S##dt##type##order, Eigen::SparseMatrix<dt>>::solve( \
      const T##dt &b);
#define EIGEN_LU_SOLVE_INSTANTIATION(dt, type, order, df)                  \
  using LUT##dt = Eigen::VectorX##df;                                      \
  using LUS##dt##type##order =                                             \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,                         \
                          Eigen::order##Ordering<int>>;                    \
  template LUT##dt                                                         \
  EigenSparseSolver<LUS##dt##type##order, Eigen::SparseMatrix<dt>>::solve( \
      const LUT##dt &b);

// Explicit instantiation of the template class EigenSparseSolver::solve_rf
#define INSTANTIATE_LLT_SOLVE_RF(dt, type, order, df)                     \
  using llt##dt##type##order =                                            \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower,      \
                              Eigen::order##Ordering<int>>;               \
  template void EigenSparseSolver<llt##dt##type##order,                   \
                                  Eigen::SparseMatrix<dt>>::solve_rf<df,  \
                                                                     dt>( \
      Program * prog, const SparseMatrix &sm, const Ndarray &b,           \
      const Ndarray &x);

#define INSTANTIATE_LU_SOLVE_RF(dt, type, order, df)                      \
  using lu##dt##type##order =                                             \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,                        \
                          Eigen::order##Ordering<int>>;                   \
  template void EigenSparseSolver<lu##dt##type##order,                    \
                                  Eigen::SparseMatrix<dt>>::solve_rf<df,  \
                                                                     dt>( \
      Program * prog, const SparseMatrix &sm, const Ndarray &b,           \
      const Ndarray &x);

#define MAKE_EIGEN_SOLVER(dt, type, order) \
  std::make_unique<EigenSparseSolver##dt##type##order>()

#define MAKE_SOLVER(dt, type, order)                              \
  {                                                               \
    {#dt, #type, #order}, []() -> std::unique_ptr<SparseSolver> { \
      return MAKE_EIGEN_SOLVER(dt, type, order);                  \
    }                                                             \
  }

using Triplets = std::tuple<std::string, std::string, std::string>;
namespace {
struct key_hash {
  std::size_t operator()(const Triplets &k) const {
    auto h1 = std::hash<std::string>{}(std::get<0>(k));
    auto h2 = std::hash<std::string>{}(std::get<1>(k));
    auto h3 = std::hash<std::string>{}(std::get<2>(k));
    return h1 ^ h2 ^ h3;
  }
};
}  // namespace

namespace taichi::lang {

#define GET_EM(sm) \
  const EigenMatrix *mat = (const EigenMatrix *)(sm.get_matrix());

template <class EigenSolver, class EigenMatrix>
bool EigenSparseSolver<EigenSolver, EigenMatrix>::compute(
    const SparseMatrix &sm) {
  if (!is_initialized_) {
    SparseSolver::init_solver(sm.num_rows(), sm.num_cols(), sm.get_data_type());
  }
  GET_EM(sm);
  solver_.compute(*mat);
  if (solver_.info() != Eigen::Success) {
    return false;
  } else
    return true;
}
template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::analyze_pattern(
    const SparseMatrix &sm) {
  if (!is_initialized_) {
    SparseSolver::init_solver(sm.num_rows(), sm.num_cols(), sm.get_data_type());
  }
  GET_EM(sm);
  solver_.analyzePattern(*mat);
}

template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::factorize(
    const SparseMatrix &sm) {
  GET_EM(sm);
  solver_.factorize(*mat);
}

template <class EigenSolver, class EigenMatrix>
template <typename T>
T EigenSparseSolver<EigenSolver, EigenMatrix>::solve(const T &b) {
  return solver_.solve(b);
}

EIGEN_LLT_SOLVE_INSTANTIATION(float32, LLT, AMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LLT, COLAMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LDLT, AMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LDLT, COLAMD, f);
EIGEN_LU_SOLVE_INSTANTIATION(float32, LU, AMD, f);
EIGEN_LU_SOLVE_INSTANTIATION(float32, LU, COLAMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LLT, AMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LLT, COLAMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LDLT, AMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LDLT, COLAMD, d);
EIGEN_LU_SOLVE_INSTANTIATION(float64, LU, AMD, d);
EIGEN_LU_SOLVE_INSTANTIATION(float64, LU, COLAMD, d);

template <class EigenSolver, class EigenMatrix>
bool EigenSparseSolver<EigenSolver, EigenMatrix>::info() {
  return solver_.info() == Eigen::Success;
}

template <class EigenSolver, class EigenMatrix>
template <typename T, typename V>
void EigenSparseSolver<EigenSolver, EigenMatrix>::solve_rf(
    Program *prog,
    const SparseMatrix &sm,
    const Ndarray &b,
    const Ndarray &x) {
  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  Eigen::Map<T>((V *)dX, rows_) = solver_.solve(Eigen::Map<T>((V *)db, cols_));
}

INSTANTIATE_LLT_SOLVE_RF(float32, LLT, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LDLT, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LLT, AMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LDLT, AMD, Eigen::VectorXf)
INSTANTIATE_LU_SOLVE_RF(float32, LU, AMD, Eigen::VectorXf)
INSTANTIATE_LU_SOLVE_RF(float32, LU, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float64, LLT, COLAMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LDLT, COLAMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LLT, AMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LDLT, AMD, Eigen::VectorXd)
INSTANTIATE_LU_SOLVE_RF(float64, LU, AMD, Eigen::VectorXd)
INSTANTIATE_LU_SOLVE_RF(float64, LU, COLAMD, Eigen::VectorXd)

CuSparseSolver::CuSparseSolver() {
#if defined(TI_WITH_CUDA)
  if (!CUSPARSEDriver::get_instance().is_loaded()) {
    bool load_success = CUSPARSEDriver::get_instance().load_cusparse();
    if (!load_success) {
      TI_ERROR("Failed to load cusparse library!");
    }
  }
  if (!CUSOLVERDriver::get_instance().is_loaded()) {
    bool load_success = CUSOLVERDriver::get_instance().load_cusolver();
    if (!load_success) {
      TI_ERROR("Failed to load cusolver library!");
    }
  }
#endif
}
// Reference:
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/cuSolverSp_LowlevelCholesky.cpp
void CuSparseSolver::analyze_pattern(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  SparseMatrix *sm_no_cv = const_cast<SparseMatrix *>(&sm);
  CuSparseMatrix *A = dynamic_cast<CuSparseMatrix *>(sm_no_cv);
  size_t rowsA = A->num_rows();
  size_t colsA = A->num_cols();
  size_t nnzA = A->get_nnz();
  void *d_csrRowPtrA = A->get_row_ptr();
  void *d_csrColIndA = A->get_col_ind();
  void *d_csrValA = A->get_val_ptr();
  CUSOLVERDriver::get_instance().csSpCreate(&cusolver_handle_);
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handel_);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descr_);
  CUSPARSEDriver::get_instance().cpSetMatType(descr_,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descr_,
                                                   CUSPARSE_INDEX_BASE_ZERO);

  // Reorder
  float *h_csrValA = NULL;
  h_Q = (int *)malloc(sizeof(int) * colsA);
  h_csrRowPtrB = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndB = (int *)malloc(sizeof(int) * nnzA);
  h_csrValB = (float *)malloc(sizeof(float) * nnzA);
  h_csrValA = (float *)malloc(sizeof(float) * nnzA);
  h_mapBfromA = (int *)malloc(sizeof(int) * nnzA);
  assert(NULL != h_Q);
  assert(NULL != h_csrRowPtrB);
  assert(NULL != h_csrColIndB);
  assert(NULL != h_csrValB);
  assert(NULL != h_mapBfromA);

  CUDADriver::get_instance().memcpy_device_to_host(h_csrRowPtrB, d_csrRowPtrA,
                                                   sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().memcpy_device_to_host(h_csrColIndB, d_csrColIndA,
                                                   sizeof(int) * nnzA);
  CUDADriver::get_instance().memcpy_device_to_host(h_csrValA, d_csrValA,
                                                   sizeof(float) * nnzA);

  // Compoute h_Q
  CUSOLVERDriver::get_instance().csSpXcsrsymamdHost(
      cusolver_handle_, rowsA, nnzA, descr_, h_csrRowPtrB, h_csrColIndB, h_Q);
  CUDADriver::get_instance().malloc((void **)&d_Q, sizeof(int) * colsA);
  CUDADriver::get_instance().memcpy_host_to_device((void *)d_Q, (void *)h_Q,
                                                   sizeof(int) * (colsA));
  size_t size_perm = 0;
  CUSOLVERDriver::get_instance().csSpXcsrperm_bufferSizeHost(
      cusolver_handle_, rowsA, colsA, nnzA, descr_, h_csrRowPtrB, h_csrColIndB,
      h_Q, h_Q, &size_perm);
  void *buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
  assert(NULL != buffer_cpu);
  for (int j = 0; j < nnzA; j++) {
    h_mapBfromA[j] = j;
  }
  CUSOLVERDriver::get_instance().csSpXcsrpermHost(
      cusolver_handle_, rowsA, colsA, nnzA, descr_, h_csrRowPtrB, h_csrColIndB,
      h_Q, h_Q, h_mapBfromA, buffer_cpu);
  /* B = A( mapBfromA ) */
  for (int j = 0; j < nnzA; j++) {
    h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
  }
  CUDADriver::get_instance().malloc((void **)&d_csrRowPtrB,
                                    sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().malloc((void **)&d_csrColIndB, sizeof(int) * nnzA);
  CUDADriver::get_instance().malloc((void **)&d_csrValB, sizeof(float) * nnzA);
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrRowPtrB, (void *)h_csrRowPtrB, sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrColIndB, (void *)h_csrColIndB, sizeof(int) * nnzA);
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrValB, (void *)h_csrValB, sizeof(float) * nnzA);
  free(h_csrValA);
  free(buffer_cpu);

  // step 1: create opaque info structure
  CUSOLVERDriver::get_instance().csSpCreateCsrcholInfo(&info_);

  // step 2: analyze chol(A) to know structure of L
  CUSOLVERDriver::get_instance().csSpXcsrcholAnalysis(
      cusolver_handle_, rowsA, nnzA, descr_, d_csrRowPtrB, d_csrColIndB, info_);
  is_analyzed_ = true;
#else
  TI_NOT_IMPLEMENTED
#endif
}

void CuSparseSolver::factorize(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  SparseMatrix *sm_no_cv = const_cast<SparseMatrix *>(&sm);
  CuSparseMatrix *A = dynamic_cast<CuSparseMatrix *>(sm_no_cv);
  size_t rowsA = A->num_rows();
  size_t nnzA = A->get_nnz();

  size_t size_internal = 0;
  size_t size_chol = 0;  // size of working space for csrlu
  // step 1: workspace for chol(A)
  CUSOLVERDriver::get_instance().csSpScsrcholBufferInfo(
      cusolver_handle_, rowsA, nnzA, descr_, d_csrValB, d_csrRowPtrB,
      d_csrColIndB, info_, &size_internal, &size_chol);

  if (size_chol > 0)
    CUDADriver::get_instance().malloc(&gpu_buffer_, sizeof(char) * size_chol);

  // step 2: compute A = L*L^T
  CUSOLVERDriver::get_instance().csSpScsrcholFactor(
      cusolver_handle_, rowsA, nnzA, descr_, d_csrValB, d_csrRowPtrB,
      d_csrColIndB, info_, gpu_buffer_);
  // step 3: check if the matrix is singular
  const float tol = 1.e-14;
  int singularity = 0;
  CUSOLVERDriver::get_instance().csSpScsrcholZeroPivot(cusolver_handle_, info_,
                                                       tol, &singularity);
  TI_ASSERT(singularity == -1);
  is_factorized_ = true;
#else
  TI_NOT_IMPLEMENTED
#endif
}

void CuSparseSolver::solve_cu(Program *prog,
                              const SparseMatrix &sm,
                              const Ndarray &b,
                              const Ndarray &x) {
#ifdef TI_WITH_CUDA
  cusparseHandle_t cusparseHandle = nullptr;
  CUSPARSEDriver::get_instance().cpCreate(&cusparseHandle);
  cusolverSpHandle_t handle = nullptr;
  CUSOLVERDriver::get_instance().csSpCreate(&handle);

  int major_version, minor_version, patch_level;
  CUSOLVERDriver::get_instance().csGetProperty(MAJOR_VERSION, &major_version);
  CUSOLVERDriver::get_instance().csGetProperty(MINOR_VERSION, &minor_version);
  CUSOLVERDriver::get_instance().csGetProperty(PATCH_LEVEL, &patch_level);
  printf("Cusolver version: %d.%d.%d\n", major_version, minor_version,
         patch_level);

  // Retrive the info of the sparse matrix
  SparseMatrix *sm_no_cv = const_cast<SparseMatrix *>(&sm);
  CuSparseMatrix *A = dynamic_cast<CuSparseMatrix *>(sm_no_cv);
  size_t nrows = A->num_rows();
  size_t ncols = A->num_cols();
  size_t nnz = A->get_nnz();
  void *drow_offsets = A->get_row_ptr();
  void *dcol_indices = A->get_col_ind();
  void *dvalues = A->get_val_ptr();

  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  size_t dx = prog->get_ndarray_data_ptr_as_int(&x);

  float *h_z = (float *)malloc(sizeof(float) * ncols);
  float *h_x = (float *)malloc(sizeof(float) * ncols);
  float *h_b = (float *)malloc(sizeof(float) * nrows);
  float *h_Qb = (float *)malloc(sizeof(float) * nrows);

  int *h_Q = (int *)malloc(sizeof(int) * ncols);
  int *hrow_offsets_B = (int *)malloc(sizeof(int) * (nrows + 1));
  int *hcol_indices_B = (int *)malloc(sizeof(int) * nnz);
  float *h_val_B = (float *)malloc(sizeof(float) * nnz);
  int *h_mapBfromA = (int *)malloc(sizeof(int) * nnz);

  assert(nullptr != h_z);
  assert(nullptr != h_x);
  assert(nullptr != h_b);
  assert(nullptr != h_Qb);
  assert(nullptr != h_Q);
  assert(nullptr != hrow_offsets_B);
  assert(nullptr != hcol_indices_B);
  assert(nullptr != h_val_B);
  assert(nullptr != h_mapBfromA);

  int *hrow_offsets = nullptr, *hcol_indices = nullptr;
  hrow_offsets = (int *)malloc(sizeof(int) * (nrows + 1));
  hcol_indices = (int *)malloc(sizeof(int) * nnz);
  assert(nullptr != hrow_offsets);
  assert(nullptr != hcol_indices);
  // Attention: drow_offsets is not freed at other palces
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)hrow_offsets, drow_offsets, sizeof(int) * (nrows + 1));
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)hcol_indices, dcol_indices, sizeof(int) * nnz);

  /* configure matrix descriptor*/
  cusparseMatDescr_t descrA = nullptr;
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrA);
  CUSPARSEDriver::get_instance().cpSetMatType(descrA,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrA,
                                                   CUSPARSE_INDEX_BASE_ZERO);
  int issym = 0;
  CUSOLVERDriver::get_instance().csSpXcsrissymHost(
      handle, nrows, nnz, descrA, (void *)hrow_offsets,
      (void *)(hrow_offsets + 1), (void *)hcol_indices, &issym);
  if (!issym) {
    TI_ERROR("A has no symmetric pattern, please use LU or QR!");
    return;
  }

  // step 2:reorder the matrix to minimize zero fill-in
  CUSOLVERDriver::get_instance().csSpXcsrsymrcmHost(
      handle, nrows, nnz, descrA, (void *)hrow_offsets, (void *)hcol_indices,
      (void *)h_Q);  // symrcm method

  // B = A(Q, Q)
  memcpy(hrow_offsets_B, hrow_offsets, sizeof(int) * (nrows + 1));
  memcpy(hcol_indices_B, hcol_indices, sizeof(int) * nnz);

  size_t size_perm;
  CUSOLVERDriver::get_instance().csSpXcsrperm_bufferSizeHost(
      handle, nrows, ncols, nnz, descrA, (void *)hrow_offsets_B,
      (void *)hcol_indices_B, (void *)h_Q, (void *)h_Q, &size_perm);
  void *buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
  for (int j = 0; j < nnz; j++)
    h_mapBfromA[j] = j;

  CUSOLVERDriver::get_instance().csSpXcsrpermHost(
      handle, nrows, ncols, nnz, descrA, (void *)hrow_offsets_B,
      (void *)hcol_indices_B, (void *)h_Q, (void *)h_Q, (void *)h_mapBfromA,
      (void *)buffer_cpu);

  float *h_val_A = (float *)malloc(sizeof(float) * nnz);
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)h_val_A, (void *)dvalues, sizeof(int) * nnz);
  for (int j = 0; j < nnz; j++)
    h_val_B[j] = h_val_A[h_mapBfromA[j]];

  CUDADriver::get_instance().memcpy_device_to_host((void *)h_b, (void *)db,
                                                   sizeof(float) * nrows);
  for (int row = 0; row < nrows; row++) {
    h_Qb[row] = h_b[h_Q[row]];
  }

  // step 4: solve B*z = Q*b
  float tol = 1e-6;
  int reorder = 1;
  int singularity = 0; /* -1 if A is invertible under tol. */
  // use Cholesky decomposition as defualt
  CUSOLVERDriver::get_instance().csSpScsrlsvcholHost(
      handle, nrows, nnz, descrA, (void *)h_val_B, (void *)hrow_offsets_B,
      (void *)hcol_indices_B, (void *)h_Qb, tol, reorder, (void *)h_z,
      &singularity);
  if (!singularity) {
    TI_ERROR("A is a sigular matrix!");
    return;
  }

  // step 5: Q*x = z
  for (int row = 0; row < nrows; row++)
    h_x[h_Q[row]] = h_z[row];

  CUDADriver::get_instance().memcpy_host_to_device((void *)dx, (void *)h_x,
                                                   sizeof(float) * ncols);

  CUSOLVERDriver::get_instance().csSpDestory(handle);
  CUSPARSEDriver::get_instance().cpDestroy(cusparseHandle);

  if (hrow_offsets != nullptr)
    free(hrow_offsets);
  if (hcol_indices != nullptr)
    free(hcol_indices);
  if (hrow_offsets_B != nullptr)
    free(hrow_offsets_B);
  if (hcol_indices_B != nullptr)
    free(hcol_indices_B);
  if (h_Q != nullptr)
    free(h_Q);
  if (h_mapBfromA != nullptr)
    free(h_mapBfromA);
  if (h_z != nullptr)
    free(h_z);
  if (h_b != nullptr)
    free(h_b);
  if (h_Qb != nullptr)
    free(h_Qb);
  if (h_x != nullptr)
    free(h_x);
  if (buffer_cpu != nullptr)
    free(buffer_cpu);
  if (h_val_A != nullptr)
    free(h_val_A);
  if (h_val_B != nullptr)
    free(h_val_B);
#endif
}

void CuSparseSolver::solve_rf(Program *prog,
                              const SparseMatrix &sm,
                              const Ndarray &b,
                              const Ndarray &x) {
#if defined(TI_WITH_CUDA)
  if (is_analyzed_ == false) {
    analyze_pattern(sm);
  }
  if (is_factorized_ == false) {
    factorize(sm);
  }
  // Retrive the info of the sparse matrix
  SparseMatrix *sm_no_cv = const_cast<SparseMatrix *>(&sm);
  CuSparseMatrix *A = dynamic_cast<CuSparseMatrix *>(sm_no_cv);
  size_t rowsA = A->num_rows();
  size_t colsA = A->num_cols();
  size_t d_b = prog->get_ndarray_data_ptr_as_int(&b);
  size_t d_x = prog->get_ndarray_data_ptr_as_int(&x);

  float *h_b = (float *)malloc(sizeof(float) * rowsA);
  float *h_Qb = (float *)malloc(sizeof(float) * rowsA);
  assert(h_b != nullptr);
  assert(h_Qb != nullptr);
  CUDADriver::get_instance().memcpy_device_to_host((void *)h_b, (void *)d_b,
                                                   sizeof(float) * rowsA);
  // replace with cuSparseGather
  for (int row = 0; row < rowsA; row++) {
    h_Qb[row] = h_b[h_Q[row]];
  }
  void *d_Qb = NULL;
  CUDADriver::get_instance().malloc(&d_Qb, sizeof(float) * rowsA);
  CUDADriver::get_instance().memcpy_host_to_device((void *)d_Qb, (void *)h_Qb,
                                                   sizeof(float) * rowsA);
  /* solve B*z = Q*b */
  void *d_z = NULL;
  CUDADriver::get_instance().malloc(&d_z, sizeof(float) * colsA);
  CUSOLVERDriver::get_instance().csSpScsrcholSolve(
      cusolver_handle_, rowsA, (void *)d_Qb, (void *)d_z, info_, gpu_buffer_);

  // Q*x = z
  CUSPARSEDriver::get_instance().cpSsctr(cusparse_handel_, (int)rowsA,
                                         (void *)d_z, (void *)d_Q, (void *)d_x,
                                         CUSPARSE_INDEX_BASE_ZERO);
  // Replace cuSparseSsctr with cuSparseScatter
  // cusparseSpVecDescr_t vecX;
  // cusparseDnVecDescr_t vecY;
  // CUSPARSEDriver::get_instance().cpCreateSpVec(&vecX, rowsA, nnz, d_Q,
  // d_z,CUSPARSE_INDEX_32I,
  //                                       CUSPARSE_INDEX_BASE_ZERO,
  //                                       CUDA_R_32F);
  // CUSPARSEDriver::get_instance().cpCreateDnVec(&vecY, rowsA, d_x,
  // CUDA_R_32F); CUSPARSEDriver::get_instance().cpScatter(cusolver_handle_,
  // vecX, vecY);

  free(h_b);
  free(h_Qb);
  CUDADriver::get_instance().mem_free(d_Qb);
  CUDADriver::get_instance().mem_free(d_z);

  // TODO: free allocated memory and handles
  // CUDADriver::get_instance().mem_free(gpu_buffer_);
  // CUSOLVERDriver::get_instance().csSpDestory(cusolver_handle_);
  // CUSPARSEDriver::get_instance().cpDestroy(cusparse_handel_);
  // CUSPARSEDriver::get_instance().cpDestroyMatDescr(descrA);
#else
  TI_NOT_IMPLEMENTED
#endif
}

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering) {
  using key_type = Triplets;
  using func_type = std::unique_ptr<SparseSolver> (*)();
  static const std::unordered_map<key_type, func_type, key_hash>
      solver_factory = {
          MAKE_SOLVER(float32, LLT, AMD),  MAKE_SOLVER(float32, LLT, COLAMD),
          MAKE_SOLVER(float32, LDLT, AMD), MAKE_SOLVER(float32, LDLT, COLAMD),
          MAKE_SOLVER(float64, LLT, AMD),  MAKE_SOLVER(float64, LLT, COLAMD),
          MAKE_SOLVER(float64, LDLT, AMD), MAKE_SOLVER(float64, LDLT, COLAMD)};
  static const std::unordered_map<std::string, std::string> dt_map = {
      {"f32", "float32"}, {"f64", "float64"}};
  auto it = dt_map.find(taichi::lang::data_type_name(dt));
  if (it == dt_map.end())
    TI_ERROR("Not supported sparse solver data type: {}",
             taichi::lang::data_type_name(dt));

  Triplets solver_key = std::make_tuple(it->second, solver_type, ordering);
  if (solver_factory.find(solver_key) != solver_factory.end()) {
    auto solver_func = solver_factory.at(solver_key);
    return solver_func();
  } else if (solver_type == "LU") {
    if (it->first == "f32") {
      using EigenMatrix = Eigen::SparseMatrix<float32>;
      using LU = Eigen::SparseLU<EigenMatrix>;
      return std::make_unique<EigenSparseSolver<LU, EigenMatrix>>();
    } else if (it->first == "f64") {
      using EigenMatrix = Eigen::SparseMatrix<float64>;
      using LU = Eigen::SparseLU<EigenMatrix>;
      return std::make_unique<EigenSparseSolver<LU, EigenMatrix>>();
    } else {
      TI_ERROR("Not supported sparse solver data type: {}", it->second);
    }
  } else
    TI_ERROR("Not supported sparse solver type: {}", solver_type);
}

CuSparseSolver::~CuSparseSolver() {
  if (h_Q != nullptr)
    free(h_Q);
  if (h_csrRowPtrB != nullptr)
    free(h_csrRowPtrB);
  if (h_csrColIndB != nullptr)
    free(h_csrColIndB);
  if (h_csrValB != nullptr)
    free(h_csrValB);
  if (h_mapBfromA != nullptr)
    free(h_mapBfromA);
  if (info_ != nullptr)
    CUSOLVERDriver::get_instance().csSpDestroyCsrcholInfo(info_);
  if (cusolver_handle_ != nullptr)
    CUSOLVERDriver::get_instance().csSpDestory(cusolver_handle_);
  if (cusparse_handel_ != nullptr)
    CUSPARSEDriver::get_instance().cpDestroy(cusparse_handel_);
  if (descr_ != nullptr)
    CUSPARSEDriver::get_instance().cpDestroyMatDescr(descr_);
  if (gpu_buffer_ != nullptr)
    CUDADriver::get_instance().mem_free(gpu_buffer_);
  if (d_Q != nullptr)
    CUDADriver::get_instance().mem_free(d_Q);
  if (d_csrRowPtrB != nullptr)
    CUDADriver::get_instance().mem_free(d_csrRowPtrB);
  if (d_csrColIndB != nullptr)
    CUDADriver::get_instance().mem_free(d_csrColIndB);
  if (d_csrValB != nullptr)
    CUDADriver::get_instance().mem_free(d_csrValB);
}
std::unique_ptr<SparseSolver> make_cusparse_solver(
    DataType dt,
    const std::string &solver_type,
    const std::string &ordering) {
  return std::make_unique<CuSparseSolver>();
}
}  // namespace taichi::lang
