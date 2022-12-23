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
  init_solver();
}

void CuSparseSolver::init_solver() {
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
void CuSparseSolver::reorder(const CuSparseMatrix &A) {
#if defined(TI_WITH_CUDA)
  size_t rowsA = A.num_rows();
  size_t colsA = A.num_cols();
  size_t nnzA = A.get_nnz();
  void *d_csrRowPtrA = A.get_row_ptr();
  void *d_csrColIndA = A.get_col_ind();
  void *d_csrValA = A.get_val_ptr();
  CUSOLVERDriver::get_instance().csSpCreate(&cusolver_handle_);
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handel_);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descr_);
  CUSPARSEDriver::get_instance().cpSetMatType(descr_,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descr_,
                                                   CUSPARSE_INDEX_BASE_ZERO);
  float *h_csrValA = nullptr;
  h_Q_ = (int *)malloc(sizeof(int) * colsA);
  h_csrRowPtrB_ = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndB_ = (int *)malloc(sizeof(int) * nnzA);
  h_csrValB_ = (float *)malloc(sizeof(float) * nnzA);
  h_csrValA = (float *)malloc(sizeof(float) * nnzA);
  h_mapBfromA_ = (int *)malloc(sizeof(int) * nnzA);
  assert(nullptr != h_Q_);
  assert(nullptr != h_csrRowPtrB_);
  assert(nullptr != h_csrColIndB_);
  assert(nullptr != h_csrValB_);
  assert(nullptr != h_mapBfromA_);

  CUDADriver::get_instance().memcpy_device_to_host(h_csrRowPtrB_, d_csrRowPtrA,
                                                   sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().memcpy_device_to_host(h_csrColIndB_, d_csrColIndA,
                                                   sizeof(int) * nnzA);
  CUDADriver::get_instance().memcpy_device_to_host(h_csrValA, d_csrValA,
                                                   sizeof(float) * nnzA);

  // compoute h_Q_
  CUSOLVERDriver::get_instance().csSpXcsrsymamdHost(cusolver_handle_, rowsA,
                                                    nnzA, descr_, h_csrRowPtrB_,
                                                    h_csrColIndB_, h_Q_);
  CUDADriver::get_instance().malloc((void **)&d_Q_, sizeof(int) * colsA);
  CUDADriver::get_instance().memcpy_host_to_device((void *)d_Q_, (void *)h_Q_,
                                                   sizeof(int) * (colsA));
  size_t size_perm = 0;
  CUSOLVERDriver::get_instance().csSpXcsrperm_bufferSizeHost(
      cusolver_handle_, rowsA, colsA, nnzA, descr_, h_csrRowPtrB_,
      h_csrColIndB_, h_Q_, h_Q_, &size_perm);
  void *buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
  assert(nullptr != buffer_cpu);
  for (int j = 0; j < nnzA; j++) {
    h_mapBfromA_[j] = j;
  }
  CUSOLVERDriver::get_instance().csSpXcsrpermHost(
      cusolver_handle_, rowsA, colsA, nnzA, descr_, h_csrRowPtrB_,
      h_csrColIndB_, h_Q_, h_Q_, h_mapBfromA_, buffer_cpu);
  // B = A( mapBfromA )
  for (int j = 0; j < nnzA; j++) {
    h_csrValB_[j] = h_csrValA[h_mapBfromA_[j]];
  }
  CUDADriver::get_instance().malloc((void **)&d_csrRowPtrB_,
                                    sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().malloc((void **)&d_csrColIndB_,
                                    sizeof(int) * nnzA);
  CUDADriver::get_instance().malloc((void **)&d_csrValB_, sizeof(float) * nnzA);
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrRowPtrB_, (void *)h_csrRowPtrB_, sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrColIndB_, (void *)h_csrColIndB_, sizeof(int) * nnzA);
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrValB_, (void *)h_csrValB_, sizeof(float) * nnzA);
  free(h_csrValA);
  free(buffer_cpu);
#endif
}

// Reference:
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/cuSolverSp_LowlevelCholesky.cpp
void CuSparseSolver::analyze_pattern(const SparseMatrix &sm) {
  switch (solver_type_) {
    case SolverType::Cholesky:
      analyze_pattern_cholesky(sm);
      break;
    case SolverType::LU:
      analyze_pattern_lu(sm);
      break;
    default:
      TI_NOT_IMPLEMENTED
  }
}
void CuSparseSolver::analyze_pattern_cholesky(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  SparseMatrix &sm_no_cv = const_cast<SparseMatrix &>(sm);
  CuSparseMatrix &A = static_cast<CuSparseMatrix &>(sm_no_cv);

  // step 1: reorder the sparse matrix
  reorder(A);

  // step 2: create opaque info structure
  CUSOLVERDriver::get_instance().csSpCreateCsrcholInfo(&info_);

  // step 3: analyze chol(A) to know structure of L
  size_t rowsA = A.num_rows();
  size_t nnzA = A.get_nnz();
  CUSOLVERDriver::get_instance().csSpXcsrcholAnalysis(
      cusolver_handle_, rowsA, nnzA, descr_, d_csrRowPtrB_, d_csrColIndB_,
      info_);
  is_analyzed_ = true;
#else
  TI_NOT_IMPLEMENTED
#endif
}
void CuSparseSolver::analyze_pattern_lu(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  SparseMatrix &sm_no_cv = const_cast<SparseMatrix &>(sm);
  CuSparseMatrix &A = static_cast<CuSparseMatrix &>(sm_no_cv);

  // step 1: reorder the sparse matrix
  reorder(A);

  // step 2: create opaque info structure
  CUSOLVERDriver::get_instance().csSpCreateCsrluInfoHost(&lu_info_);

  // step 3: analyze LU(B) to know structure of Q and R, and upper bound for
  // nnz(L+U)
  size_t rowsA = A.num_rows();
  size_t nnzA = A.get_nnz();
  CUSOLVERDriver::get_instance().csSpXcsrluAnalysisHost(
      cusolver_handle_, rowsA, nnzA, descr_, h_csrRowPtrB_, h_csrColIndB_,
      lu_info_);
  is_analyzed_ = true;
#else
  TI_NOT_IMPLEMENTED
#endif
}
void CuSparseSolver::factorize(const SparseMatrix &sm) {
  switch (solver_type_) {
    case SolverType::Cholesky:
      factorize_cholesky(sm);
      break;
    case SolverType::LU:
      factorize_lu(sm);
      break;
    default:
      TI_NOT_IMPLEMENTED
  }
}
void CuSparseSolver::factorize_cholesky(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  SparseMatrix *sm_no_cv = const_cast<SparseMatrix *>(&sm);
  CuSparseMatrix *A = static_cast<CuSparseMatrix *>(sm_no_cv);
  size_t rowsA = A->num_rows();
  size_t nnzA = A->get_nnz();

  size_t size_internal = 0;
  size_t size_chol = 0;  // size of working space for csrlu
  // step 1: workspace for chol(A)
  CUSOLVERDriver::get_instance().csSpScsrcholBufferInfo(
      cusolver_handle_, rowsA, nnzA, descr_, d_csrValB_, d_csrRowPtrB_,
      d_csrColIndB_, info_, &size_internal, &size_chol);

  if (size_chol > 0)
    CUDADriver::get_instance().malloc(&gpu_buffer_, sizeof(char) * size_chol);

  // step 2: compute A = L*L^T
  CUSOLVERDriver::get_instance().csSpScsrcholFactor(
      cusolver_handle_, rowsA, nnzA, descr_, d_csrValB_, d_csrRowPtrB_,
      d_csrColIndB_, info_, gpu_buffer_);
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
void CuSparseSolver::factorize_lu(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  SparseMatrix *sm_no_cv = const_cast<SparseMatrix *>(&sm);
  CuSparseMatrix *A = static_cast<CuSparseMatrix *>(sm_no_cv);
  size_t rowsA = A->num_rows();
  size_t nnzA = A->get_nnz();
  // step 4: workspace for LU(B)
  size_t size_lu = 0;
  size_t buffer_size = 0;
  CUSOLVERDriver::get_instance().csSpScsrluBufferInfoHost(
      cusolver_handle_, rowsA, nnzA, descr_, h_csrValB_, h_csrRowPtrB_,
      h_csrColIndB_, lu_info_, &buffer_size, &size_lu);

  if (cpu_buffer_)
    free(cpu_buffer_);
  cpu_buffer_ = (void *)malloc(sizeof(char) * size_lu);
  assert(nullptr != cpu_buffer_);

  // step 5: compute Ppivot * B = L * U
  CUSOLVERDriver::get_instance().csSpScsrluFactorHost(
      cusolver_handle_, rowsA, nnzA, descr_, h_csrValB_, h_csrRowPtrB_,
      h_csrColIndB_, lu_info_, 1.0f, cpu_buffer_);

  // step 6: check singularity by tol
  int singularity = 0;
  const float tol = 1.e-6;
  CUSOLVERDriver::get_instance().csSpScsrluZeroPivotHost(
      cusolver_handle_, lu_info_, tol, &singularity);
  TI_ASSERT(singularity == -1);
  is_factorized_ = true;
#else
  TI_NOT_IMPLEMENTED
#endif
}
void CuSparseSolver::solve_rf(Program *prog,
                              const SparseMatrix &sm,
                              const Ndarray &b,
                              const Ndarray &x) {
  switch (solver_type_) {
    case SolverType::Cholesky:
      solve_cholesky(prog, sm, b, x);
      break;
    case SolverType::LU:
      solve_lu(prog, sm, b, x);
      break;
    default:
      TI_NOT_IMPLEMENTED
  }
}

void CuSparseSolver::solve_cholesky(Program *prog,
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
  CuSparseMatrix *A = static_cast<CuSparseMatrix *>(sm_no_cv);
  size_t rowsA = A->num_rows();
  size_t colsA = A->num_cols();
  size_t d_b = prog->get_ndarray_data_ptr_as_int(&b);
  size_t d_x = prog->get_ndarray_data_ptr_as_int(&x);

  // step 1: d_Qb = Q * b
  void *d_Qb = nullptr;
  CUDADriver::get_instance().malloc(&d_Qb, sizeof(float) * rowsA);
  cusparseDnVecDescr_t vec_b;
  cusparseSpVecDescr_t vec_Qb;
  CUSPARSEDriver::get_instance().cpCreateDnVec(&vec_b, (int)rowsA, (void *)d_b,
                                               CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpCreateSpVec(
      &vec_Qb, (int)rowsA, (int)rowsA, d_Q_, d_Qb, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpGather(cusparse_handel_, vec_b, vec_Qb);

  // step 2: solve B*z = Q*b using cholesky solver
  void *d_z = nullptr;
  CUDADriver::get_instance().malloc(&d_z, sizeof(float) * colsA);
  CUSOLVERDriver::get_instance().csSpScsrcholSolve(
      cusolver_handle_, rowsA, (void *)d_Qb, (void *)d_z, info_, gpu_buffer_);

  // step 3: Q*x = z
  cusparseSpVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  CUSPARSEDriver::get_instance().cpCreateSpVec(
      &vecX, (int)colsA, (int)colsA, d_Q_, d_z, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpCreateDnVec(&vecY, (int)colsA, (void *)d_x,
                                               CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpScatter(cusparse_handel_, vecX, vecY);

  if (d_Qb != nullptr)
    CUDADriver::get_instance().mem_free(d_Qb);
  if (d_z != nullptr)
    CUDADriver::get_instance().mem_free(d_z);
  CUSPARSEDriver::get_instance().cpDestroySpVec(vec_Qb);
  CUSPARSEDriver::get_instance().cpDestroyDnVec(vec_b);
  CUSPARSEDriver::get_instance().cpDestroySpVec(vecX);
  CUSPARSEDriver::get_instance().cpDestroyDnVec(vecY);
#else
  TI_NOT_IMPLEMENTED
#endif
}

void CuSparseSolver::solve_lu(Program *prog,
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
  CuSparseMatrix *A = static_cast<CuSparseMatrix *>(sm_no_cv);
  size_t rowsA = A->num_rows();
  size_t colsA = A->num_cols();

  // step 7: solve L*U*x = b
  size_t d_b = prog->get_ndarray_data_ptr_as_int(&b);
  size_t d_x = prog->get_ndarray_data_ptr_as_int(&x);
  float *h_b = (float *)malloc(sizeof(float) * rowsA);
  float *h_b_hat = (float *)malloc(sizeof(float) * rowsA);
  float *h_x = (float *)malloc(sizeof(float) * colsA);
  float *h_x_hat = (float *)malloc(sizeof(float) * colsA);
  assert(nullptr != h_b);
  assert(nullptr != h_b_hat);
  assert(nullptr != h_x);
  assert(nullptr != h_x_hat);
  CUDADriver::get_instance().memcpy_device_to_host((void *)h_b, (void *)d_b,
                                                   sizeof(float) * rowsA);
  CUDADriver::get_instance().memcpy_device_to_host((void *)h_x, (void *)d_x,
                                                   sizeof(float) * colsA);
  for (int j = 0; j < rowsA; j++) {
    h_b_hat[j] = h_b[h_Q_[j]];
  }
  CUSOLVERDriver::get_instance().csSpScsrluSolveHost(
      cusolver_handle_, rowsA, h_b_hat, h_x_hat, lu_info_, cpu_buffer_);
  for (int j = 0; j < colsA; j++) {
    h_x[h_Q_[j]] = h_x_hat[j];
  }
  CUDADriver::get_instance().memcpy_host_to_device((void *)d_x, (void *)h_x,
                                                   sizeof(float) * colsA);

  free(h_b);
  free(h_b_hat);
  free(h_x);
  free(h_x_hat);
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
#if defined(TI_WITH_CUDA)
  if (h_Q_ != nullptr)
    free(h_Q_);
  if (h_csrRowPtrB_ != nullptr)
    free(h_csrRowPtrB_);
  if (h_csrColIndB_ != nullptr)
    free(h_csrColIndB_);
  if (h_csrValB_ != nullptr)
    free(h_csrValB_);
  if (h_mapBfromA_ != nullptr)
    free(h_mapBfromA_);
  if (cpu_buffer_ != nullptr)
    free(cpu_buffer_);
  if (info_ != nullptr)
    CUSOLVERDriver::get_instance().csSpDestroyCsrcholInfo(info_);
  if (lu_info_ != nullptr)
    CUSOLVERDriver::get_instance().csSpDestroyCsrluInfoHost(lu_info_);
  if (cusolver_handle_ != nullptr)
    CUSOLVERDriver::get_instance().csSpDestory(cusolver_handle_);
  if (cusparse_handel_ != nullptr)
    CUSPARSEDriver::get_instance().cpDestroy(cusparse_handel_);
  if (descr_ != nullptr)
    CUSPARSEDriver::get_instance().cpDestroyMatDescr(descr_);
  if (gpu_buffer_ != nullptr)
    CUDADriver::get_instance().mem_free(gpu_buffer_);
  if (d_Q_ != nullptr)
    CUDADriver::get_instance().mem_free(d_Q_);
  if (d_csrRowPtrB_ != nullptr)
    CUDADriver::get_instance().mem_free(d_csrRowPtrB_);
  if (d_csrColIndB_ != nullptr)
    CUDADriver::get_instance().mem_free(d_csrColIndB_);
  if (d_csrValB_ != nullptr)
    CUDADriver::get_instance().mem_free(d_csrValB_);
#endif
}
std::unique_ptr<SparseSolver> make_cusparse_solver(
    DataType dt,
    const std::string &solver_type,
    const std::string &ordering) {
  if (solver_type == "LLT" || solver_type == "LDLT") {
    return std::make_unique<CuSparseSolver>(
        CuSparseSolver::SolverType::Cholesky);
  } else if (solver_type == "LU") {
    return std::make_unique<CuSparseSolver>(CuSparseSolver::SolverType::LU);
  } else {
    TI_ERROR("Not supported sparse solver type: {}", solver_type);
  }
}
}  // namespace taichi::lang
