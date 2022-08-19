#include "taichi/ir/type_utils.h"

#include "sparse_solver.h"

#include <unordered_map>

#define MAKE_SOLVER(dt, type, order)                                           \
  {                                                                            \
    {#dt, #type, #order}, []() -> std::unique_ptr<SparseSolver> {              \
      using T = Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                                        Eigen::order##Ordering<int>>;          \
      return std::make_unique<                                                 \
          EigenSparseSolver<T, Eigen::SparseMatrix<dt>>>();                    \
    }                                                                          \
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

namespace taichi {
namespace lang {

#define GET_EM(sm) \
  const EigenMatrix *mat = (const EigenMatrix *)(sm.get_matrix());

template <class EigenSolver, class EigenMatrix>
bool EigenSparseSolver<EigenSolver, EigenMatrix>::compute(
    const SparseMatrix &sm) {
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
Eigen::VectorXf EigenSparseSolver<EigenSolver, EigenMatrix>::solve(
    const Eigen::Ref<const Eigen::VectorXf> &b) {
  return solver_.solve(b);
}

template <class EigenSolver, class EigenMatrix>
bool EigenSparseSolver<EigenSolver, EigenMatrix>::info() {
  return solver_.info() == Eigen::Success;
}

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering) {
  using key_type = Triplets;
  using func_type = std::unique_ptr<SparseSolver> (*)();
  static const std::unordered_map<key_type, func_type, key_hash>
      solver_factory = {
          MAKE_SOLVER(float32, LLT, AMD), MAKE_SOLVER(float32, LLT, COLAMD),
          MAKE_SOLVER(float32, LDLT, AMD), MAKE_SOLVER(float32, LDLT, COLAMD)};
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
    using EigenMatrix = Eigen::SparseMatrix<float32>;
    using LU = Eigen::SparseLU<EigenMatrix>;
    return std::make_unique<EigenSparseSolver<LU, EigenMatrix>>();
  } else
    TI_ERROR("Not supported sparse solver type: {}", solver_type);
}

std::unique_ptr<SparseSolver> make_cusparse_solver(
    DataType dt,
    const std::string &solver_type,
    const std::string &ordering) {
#if defined(TI_WITH_CUDA)
  if (!CUSOLVERDriver::get_instance().is_loaded()) {
    bool load_success = CUSOLVERDriver::get_instance().load_cusolver();
    if (!load_success) {
      TI_ERROR("Failed to load cusolver library!");
    }
  }
  int major_version, minor_version, patch_level;
  CUSOLVERDriver::get_instance().csGetProperty(MAJOR_VERSION, &major_version);
  CUSOLVERDriver::get_instance().csGetProperty(MINOR_VERSION, &minor_version);
  CUSOLVERDriver::get_instance().csGetProperty(PATCH_LEVEL, &patch_level);
  printf("CUDA solver version: %d.%d.%d\n", major_version, minor_version,
         patch_level);
#endif
  return nullptr;
}

void cu_solve(Program *prog,
              const Ndarray &row_offsets,
              const Ndarray &col_indices,
              const Ndarray &values,
              int nrows,
              int ncols,
              int nnz,
              const Ndarray &b,
              Ndarray &x) {
#if defined(TI_WITH_CUDA)
  size_t drow_offsets = prog->get_ndarray_data_ptr_as_int(&row_offsets);
  size_t dcol_indices = prog->get_ndarray_data_ptr_as_int(&col_indices);
  size_t dvalues = prog->get_ndarray_data_ptr_as_int(&values);
  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  size_t dx = prog->get_ndarray_data_ptr_as_int(&x);

  if (!CUSOLVERDriver::get_instance().is_loaded()) {
    bool load_success = CUSOLVERDriver::get_instance().load_cusolver();
    if (!load_success) {
      TI_ERROR("Failed to load cusolver library!");
    }
  }
  int major_version, minor_version, patch_level;
  CUSOLVERDriver::get_instance().csGetProperty(MAJOR_VERSION, &major_version);
  CUSOLVERDriver::get_instance().csGetProperty(MINOR_VERSION, &minor_version);
  CUSOLVERDriver::get_instance().csGetProperty(PATCH_LEVEL, &patch_level);
  printf("CUDA solver version: %d.%d.%d\n", major_version, minor_version,
         patch_level);

  cusolverSpHandle_t handle = NULL;
  cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
  cusparseMatDescr_t descrA = NULL;
  CUSOLVERDriver::get_instance().csSpCreate(&handle);
  CUSPARSEDriver::get_instance().cpCreate(&cusparseHandle);

  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrA);
  CUSPARSEDriver::get_instance().cpSetMatType(descrA,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrA,
                                                   CUSPARSE_INDEX_BASE_ZERO);

  int *hrow_offsets = NULL, *hcol_indices = NULL;
  hrow_offsets = (int *)malloc(sizeof(int) * (nrows + 1));
  hcol_indices = (int *)malloc(sizeof(int) * nnz);
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)hrow_offsets, (void *)drow_offsets, sizeof(int) * (nrows + 1));
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)hcol_indices, (void *)dcol_indices, sizeof(int) * nnz);

  int issym = 0;
  CUSOLVERDriver::get_instance().csSpXcsrissymHost(
      handle, nrows, nnz, descrA, (void *)hrow_offsets,
      (void *)(hrow_offsets + 1), (void *)hcol_indices, &issym);
  if (!issym) {
    printf("Error: A has no symmetric pattern, please use LU or QR \n");
    return;
  }

  // step 2:reorder the matrix to minimize zero fill-in
  int *h_Q = (int *)malloc(sizeof(int) * ncols);
  CUSOLVERDriver::get_instance().csSpXcsrsymrcmHost(
      handle, nrows, nnz, descrA, (void *)hrow_offsets, (void *)hcol_indices,
      (void *)h_Q);  // symrcm method

  // B = A(Q, Q)
  int *hrow_offsets_B = NULL, *hcol_indices_B = NULL;
  hrow_offsets_B = (int *)malloc(sizeof(int) * (nrows + 1));
  hcol_indices_B = (int *)malloc(sizeof(int) * nnz);
  memcpy(hrow_offsets_B, hrow_offsets, sizeof(int) * (nrows + 1));
  memcpy(hcol_indices_B, hcol_indices, sizeof(int) * nnz);
  size_t size_perm;
  CUSOLVERDriver::get_instance().csSpXcsrperm_bufferSizeHost(
      handle, nrows, ncols, nnz, descrA, (void *)hrow_offsets_B,
      (void *)hcol_indices_B, (void *)h_Q, (void *)h_Q, &size_perm);

  void *buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
  int *h_mapBfromA = (int *)malloc(sizeof(int) * nnz);
  for (int j = 0; j < nnz; j++)
    h_mapBfromA[j] = j;
  CUSOLVERDriver::get_instance().csSpXcsrpermHost(
      handle, nrows, ncols, nnz, descrA, (void *)hrow_offsets_B,
      (void *)hcol_indices_B, (void *)h_Q, (void *)h_Q, (void *)h_mapBfromA,
      (void *)buffer_cpu);

  // B = A(mapBfromA)
  float *h_val_A = (float *)malloc(sizeof(float) * nnz);
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)h_val_A, (void *)dvalues, sizeof(int) * nnz);
  float *h_val_B = (float *)malloc(sizeof(float) * nnz);
  if (h_val_B == NULL)
    TI_ERROR("Failed to allocate memory for h_val_B");
  for (int j = 0; j < nnz; j++)
    h_val_B[j] = h_val_A[h_mapBfromA[j]];

  // step 3: b(j) = 1 + j/n
  float *h_b = (float *)malloc(sizeof(float) * nrows);
  CUDADriver::get_instance().memcpy_device_to_host((void *)h_b, (void *)db,
                                                   sizeof(float) * nrows);

  float *h_Qb = (float *)malloc(sizeof(float) * nrows);
  for (int row = 0; row < nrows; row++) {
    h_Qb[row] = h_b[h_Q[row]];
  }

  // step 4: solve B*z = Q*b
  float *h_z = (float *)malloc(sizeof(float) * ncols);
  float tol = 1e-6;
  int reorder = 0;
  int singularity = 0; /* -1 if A is invertible under tol. */
  CUSOLVERDriver::get_instance().csSpScsrlsvcholHost(
      handle, nrows, nnz, descrA, (void *)h_val_B, (void *)hrow_offsets_B,
      (void *)hcol_indices_B, (void *)h_Qb, tol, reorder, (void *)h_z,
      &singularity);

  // step 5: Q*x = z
  float *h_x = (float *)malloc(sizeof(float) * ncols);
  for (int row = 0; row < nrows; row++)
    h_x[h_Q[row]] = h_z[row];

  for (int col = ncols - 10; col < ncols; col++)
    printf("x[%d] = %f\n", col, h_x[col]);

  CUDADriver::get_instance().memcpy_host_to_device((void *)dx, (void *)h_x,
                                                   sizeof(float) * ncols);

  CUSOLVERDriver::get_instance().csSpDestory(handle);
  CUSPARSEDriver::get_instance().cpDestroy(cusparseHandle);

  if (hrow_offsets != NULL)
    free(hrow_offsets);
  if (hcol_indices != NULL)
    free(hcol_indices);
  if (h_Q != NULL)
    free(h_Q);
  if (buffer_cpu != NULL)
    free(buffer_cpu);
  if (h_mapBfromA != NULL)
    free(h_mapBfromA);
  if (h_val_A != NULL)
    free(h_val_A);
  if (h_b != NULL)
    free(h_b);
  if (h_Qb != NULL)
    free(h_Qb);
  if (h_x != NULL)
    free(h_x);
#endif
}

}  // namespace lang
}  // namespace taichi
