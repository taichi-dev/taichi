#include "taichi/program/sparse_matrix.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "Eigen/Dense"
#include "Eigen/SparseLU"

#define BUILD(TYPE)                                                         \
  {                                                                         \
    using T = Eigen::Triplet<float##TYPE>;                                  \
    std::vector<T> *triplets = static_cast<std::vector<T> *>(triplets_adr); \
    matrix_.setFromTriplets(triplets->begin(), triplets->end());            \
  }

#define MAKE_MATRIX(TYPE, STORAGE)                                             \
  {                                                                            \
    Pair("f" #TYPE, #STORAGE),                                                 \
        [](int rows, int cols, DataType dt) -> std::unique_ptr<SparseMatrix> { \
          using FC = Eigen::SparseMatrix<float##TYPE, Eigen::STORAGE>;         \
          return std::make_unique<EigenSparseMatrix<FC>>(rows, cols, dt);      \
        }                                                                      \
  }

namespace {
using Pair = std::pair<std::string, std::string>;
struct key_hash {
  std::size_t operator()(const Pair &k) const {
    auto h1 = std::hash<std::string>{}(k.first);
    auto h2 = std::hash<std::string>{}(k.second);
    return h1 ^ h2;
  }
};

template <typename T, typename T1, typename T2>
void print_triplet_from_csr(int64_t n_rows,
                            int n_cols,
                            T *row,
                            T1 *col,
                            T2 *value,
                            std::ostringstream &ostr) {
  using Triplets = Eigen::Triplet<T2>;
  std::vector<Triplets> trips;
  for (int64_t i = 1; i <= n_rows; ++i) {
    auto n_i = row[i] - row[i - 1];
    for (auto j = 0; j < n_i; ++j) {
      trips.push_back({static_cast<int>(i - 1),
                       static_cast<int>(col[row[i - 1] + j]),
                       static_cast<float>(value[row[i - 1] + j])});
    }
  }
  Eigen::SparseMatrix<float> m(n_rows, n_cols);
  m.setFromTriplets(trips.begin(), trips.end());
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  ostr << Eigen::MatrixXf(m.cast<float>()).format(clean_fmt);
}

}  // namespace

namespace taichi::lang {

SparseMatrixBuilder::SparseMatrixBuilder(int rows,
                                         int cols,
                                         int max_num_triplets,
                                         DataType dtype,
                                         const std::string &storage_format,
                                         Program *prog)
    : rows_(rows),
      cols_(cols),
      max_num_triplets_(max_num_triplets),
      dtype_(dtype),
      storage_format_(storage_format),
      prog_(prog) {
  auto element_size = data_type_size(dtype);
  TI_ASSERT((element_size == 4 || element_size == 8));
  data_base_ptr_ndarray_ = std::make_unique<Ndarray>(
      prog_, dtype_, std::vector<int>{3 * (int)max_num_triplets_});
}

template <typename T, typename G>
void SparseMatrixBuilder::print_template() {
  fmt::print("n={}, m={}, num_triplets={} (max={})\n", rows_, cols_,
             num_triplets_, max_num_triplets_);
  T *data = reinterpret_cast<T *>(data_base_ptr_.get());
  for (int64 i = 0; i < num_triplets_; i++) {
    fmt::print("({}, {}) val={}\n", ((G *)data)[i * 3], ((G *)data)[i * 3 + 1],
               taichi_union_cast<T>(data[i * 3 + 2]));
  }
  fmt::print("\n");
}

void SparseMatrixBuilder::print_triplets() {
  num_triplets_ = data_base_ptr_ndarray_->read_int(std::vector<int>{0});
  fmt::print("n={}, m={}, num_triplets={} (max={})\n", rows_, cols_,
             num_triplets_, max_num_triplets_);
  for (int i = 0; i < num_triplets_; i++) {
    auto idx = 3 * i + 1;
    auto row = data_base_ptr_ndarray_->read_int(std::vector<int>{idx});
    auto col = data_base_ptr_ndarray_->read_int(std::vector<int>{idx + 1});
    auto val = data_base_ptr_ndarray_->read_float(std::vector<int>{idx + 2});
    fmt::print("[{}, {}] = {}\n", row, col, val);
  }
}

intptr_t SparseMatrixBuilder::get_ndarray_data_ptr() const {
  return prog_->get_ndarray_data_ptr_as_int(data_base_ptr_ndarray_.get());
}

template <typename T, typename G>
void SparseMatrixBuilder::build_template(std::unique_ptr<SparseMatrix> &m) {
  using V = Eigen::Triplet<T>;
  std::vector<V> triplets;
  auto ptr = get_ndarray_data_ptr();
  G *data = reinterpret_cast<G *>(ptr);
  num_triplets_ = data[0];
  data += 1;
  for (int i = 0; i < num_triplets_; i++) {
    triplets.push_back(
        V(data[i * 3], data[i * 3 + 1], taichi_union_cast<T>(data[i * 3 + 2])));
    fmt::print("({}, {}) val={}\n", data[i * 3], data[i * 3 + 1],
               taichi_union_cast<T>(data[i * 3 + 2]));
  }
  m->build_triplets(static_cast<void *>(&triplets));
  clear();
}

std::unique_ptr<SparseMatrix> SparseMatrixBuilder::build() {
  TI_ASSERT(built_ == false);
  built_ = true;
  auto sm = make_sparse_matrix(rows_, cols_, dtype_, storage_format_);
  auto element_size = data_type_size(dtype_);
  switch (element_size) {
    case 4:
      build_template<float32, int32>(sm);
      break;
    case 8:
      build_template<float64, int64>(sm);
      break;
    default:
      TI_ERROR("Unsupported sparse matrix data type!");
      break;
  }
  return sm;
}

void SparseMatrixBuilder::clear() {
  built_ = false;
  num_triplets_ = 0;
}

template <class EigenMatrix>
const std::string EigenSparseMatrix<EigenMatrix>::to_string() const {
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  // Note that the code below first converts the sparse matrix into a dense one.
  // https://stackoverflow.com/questions/38553335/how-can-i-print-in-console-a-formatted-sparse-matrix-with-eigen
  std::ostringstream ostr;
  ostr << Eigen::MatrixXf(matrix_.template cast<float>()).format(clean_fmt);
  return ostr.str();
}

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::build_triplets(void *triplets_adr) {
  std::string sdtype = taichi::lang::data_type_name(dtype_);
  if (sdtype == "f32") {
    BUILD(32)
  } else if (sdtype == "f64") {
    BUILD(64)
  } else {
    TI_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
}

std::unique_ptr<SparseMatrix> make_sparse_matrix(
    int rows,
    int cols,
    DataType dt,
    const std::string &storage_format = "col_major") {
  using func_type = std::unique_ptr<SparseMatrix> (*)(int, int, DataType);
  static const std::unordered_map<Pair, func_type, key_hash> map = {
      MAKE_MATRIX(32, ColMajor), MAKE_MATRIX(32, RowMajor),
      MAKE_MATRIX(64, ColMajor), MAKE_MATRIX(64, RowMajor)};
  std::unordered_map<std::string, std::string> format_map = {
      {"col_major", "ColMajor"}, {"row_major", "RowMajor"}};
  std::string tdt = taichi::lang::data_type_name(dt);
  Pair key = std::make_pair(tdt, format_map.at(storage_format));
  auto it = map.find(key);
  if (it != map.end()) {
    auto func = map.at(key);
    return func(rows, cols, dt);
  } else
    TI_ERROR("Unsupported sparse matrix data type: {}, storage format: {}", tdt,
             storage_format);
}

std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(int rows,
                                                    int cols,
                                                    DataType dt) {
  return std::unique_ptr<SparseMatrix>(
      std::make_unique<CuSparseMatrix>(rows, cols, dt));
}

std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(cusparseSpMatDescr_t mat,
                                                    int rows,
                                                    int cols,
                                                    DataType dt) {
  return std::unique_ptr<SparseMatrix>(
      std::make_unique<CuSparseMatrix>(mat, rows, cols, dt));
}

template <typename T>
void build_ndarray_template(SparseMatrix &sm,
                            intptr_t data_ptr,
                            size_t num_triplets) {
  using V = Eigen::Triplet<T>;
  std::vector<V> triplets;
  T *data = reinterpret_cast<T *>(data_ptr);
  for (int i = 0; i < num_triplets; i++) {
    triplets.push_back(
        V(data[i * 3], data[i * 3 + 1], taichi_union_cast<T>(data[i * 3 + 2])));
  }
  sm.build_triplets(static_cast<void *>(&triplets));
}

void make_sparse_matrix_from_ndarray(Program *prog,
                                     SparseMatrix &sm,
                                     const Ndarray &ndarray) {
  std::string sdtype = taichi::lang::data_type_name(sm.get_data_type());
  auto data_ptr = prog->get_ndarray_data_ptr_as_int(&ndarray);
  auto num_triplets = ndarray.get_nelement() * ndarray.get_element_size() / 3;
  if (sdtype == "f32") {
    build_ndarray_template<float32>(sm, data_ptr, num_triplets);
  } else if (sdtype == "f64") {
    build_ndarray_template<float64>(sm, data_ptr, num_triplets);
  } else {
    TI_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
}

void CuSparseMatrix::build_csr_from_coo(void *coo_row_ptr,
                                        void *coo_col_ptr,
                                        void *coo_values_ptr,
                                        int nnz) {
#if defined(TI_WITH_CUDA)
  // Step 1: Sort coo first
  cusparseHandle_t cusparse_handle = nullptr;
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handle);
  cusparseSpVecDescr_t vec_permutation;
  cusparseDnVecDescr_t vec_values;
  void *d_permutation = nullptr, *d_values_sorted = nullptr;
  CUDADriver::get_instance().malloc(&d_permutation, nnz * sizeof(int));
  CUDADriver::get_instance().malloc(&d_values_sorted, nnz * sizeof(float));
  CUSPARSEDriver::get_instance().cpCreateSpVec(
      &vec_permutation, nnz, nnz, d_permutation, d_values_sorted,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpCreateDnVec(&vec_values, nnz, coo_values_ptr,
                                               CUDA_R_32F);
  size_t bufferSize = 0;
  CUSPARSEDriver::get_instance().cpXcoosort_bufferSizeExt(
      cusparse_handle, rows_, cols_, nnz, coo_row_ptr, coo_col_ptr,
      &bufferSize);
  void *dbuffer = nullptr;
  if (bufferSize > 0)
    CUDADriver::get_instance().malloc(&dbuffer, bufferSize);
  // Setup permutation vector to identity
  CUSPARSEDriver::get_instance().cpCreateIdentityPermutation(
      cusparse_handle, nnz, d_permutation);
  CUSPARSEDriver::get_instance().cpXcoosortByRow(cusparse_handle, rows_, cols_,
                                                 nnz, coo_row_ptr, coo_col_ptr,
                                                 d_permutation, dbuffer);
  CUSPARSEDriver::get_instance().cpGather(cusparse_handle, vec_values,
                                          vec_permutation);
  CUDADriver::get_instance().memcpy_device_to_device(
      coo_values_ptr, d_values_sorted, nnz * sizeof(float));
  // Step 2: coo to csr
  void *csr_row_offset_ptr = nullptr;
  CUDADriver::get_instance().malloc(&csr_row_offset_ptr,
                                    sizeof(int) * (rows_ + 1));
  CUSPARSEDriver::get_instance().cpCoo2Csr(
      cusparse_handle, (void *)coo_row_ptr, nnz, rows_,
      (void *)csr_row_offset_ptr, CUSPARSE_INDEX_BASE_ZERO);

  CUSPARSEDriver::get_instance().cpCreateCsr(
      &matrix_, rows_, cols_, nnz, csr_row_offset_ptr, coo_col_ptr,
      coo_values_ptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpDestroySpVec(vec_permutation);
  CUSPARSEDriver::get_instance().cpDestroyDnVec(vec_values);
  CUSPARSEDriver::get_instance().cpDestroy(cusparse_handle);
  // TODO: free csr_row_offset_ptr
  // CUDADriver::get_instance().mem_free(csr_row_offset_ptr);
  CUDADriver::get_instance().mem_free(d_values_sorted);
  CUDADriver::get_instance().mem_free(d_permutation);
  CUDADriver::get_instance().mem_free(dbuffer);
  csr_row_ptr_ = csr_row_offset_ptr;
  csr_col_ind_ = coo_col_ptr;
  csr_val_ = coo_values_ptr;
  nnz_ = nnz;
#endif
}

CuSparseMatrix::~CuSparseMatrix() {
#if defined(TI_WITH_CUDA)
  CUSPARSEDriver::get_instance().cpDestroySpMat(matrix_);
#endif
}
void make_sparse_matrix_from_ndarray_cusparse(Program *prog,
                                              SparseMatrix &sm,
                                              const Ndarray &row_coo,
                                              const Ndarray &col_coo,
                                              const Ndarray &val_coo) {
#if defined(TI_WITH_CUDA)
  size_t coo_row_ptr = prog->get_ndarray_data_ptr_as_int(&row_coo);
  size_t coo_col_ptr = prog->get_ndarray_data_ptr_as_int(&col_coo);
  size_t coo_val_ptr = prog->get_ndarray_data_ptr_as_int(&val_coo);
  int nnz = val_coo.get_nelement();
  sm.build_csr_from_coo((void *)coo_row_ptr, (void *)coo_col_ptr,
                        (void *)coo_val_ptr, nnz);
#endif
}

// Reference::https://docs.nvidia.com/cuda/cusparse/index.html#csrgeam2
std::unique_ptr<SparseMatrix> CuSparseMatrix::addition(
    const CuSparseMatrix &other,
    const float alpha,
    const float beta) const {
#if defined(TI_WITH_CUDA)
  // Get information of this matrix: A
  size_t nrows_A = 0, ncols_A = 0, nnz_A = 0;
  void *drow_offsets_A = nullptr, *dcol_indices_A = nullptr,
       *dvalues_A = nullptr;
  cusparseIndexType_t csrRowOffsetsType_A, csrColIndType_A;
  cusparseIndexBase_t idxBase_A;
  cudaDataType valueType_A;
  TI_ASSERT(matrix_ != nullptr);

  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &nrows_A, &ncols_A, &nnz_A, &drow_offsets_A, &dcol_indices_A,
      &dvalues_A, &csrRowOffsetsType_A, &csrColIndType_A, &idxBase_A,
      &valueType_A);
  // Get information of other matrix: B
  size_t nrows_B = 0, ncols_B = 0, nnz_B = 0;
  void *drow_offsets_B = nullptr, *dcol_indices_B = nullptr,
       *dvalues_B = nullptr;
  cusparseIndexType_t csrRowOffsetsType_B, csrColIndType_B;
  cusparseIndexBase_t idxBase_B;
  cudaDataType valueType_B;
  CUSPARSEDriver::get_instance().cpCsrGet(
      other.matrix_, &nrows_B, &ncols_B, &nnz_B, &drow_offsets_B,
      &dcol_indices_B, &dvalues_B, &csrRowOffsetsType_B, &csrColIndType_B,
      &idxBase_B, &valueType_B);

  // Create sparse matrix: C
  int *drow_offsets_C = nullptr;
  int *dcol_indices_C = nullptr;
  float *dvalues_C = nullptr;
  cusparseMatDescr_t descrA = nullptr, descrB = nullptr, descrC = nullptr;
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrA);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrB);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrC);
  CUSPARSEDriver::get_instance().cpSetMatType(descrA,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatType(descrB,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatType(descrC,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrC,
                                                   CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrA,
                                                   CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrB,
                                                   CUSPARSE_INDEX_BASE_ZERO);

  // Start to do addition
  cusparseHandle_t cusparse_handle;
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handle);
  // alpha, nnzTotalDevHostPtr points to host memory
  size_t BufferSizeInBytes;
  char *buffer = nullptr;
  int nnzC;
  int *nnzTotalDevHostPtr = &nnzC;
  CUSPARSEDriver::get_instance().cpSetPointerMode(cusparse_handle,
                                                  CUSPARSE_POINTER_MODE_HOST);
  CUDADriver::get_instance().malloc((void **)(&drow_offsets_C),
                                    sizeof(int) * (nrows_A + 1));
  // Prepare buffer
  CUSPARSEDriver::get_instance().cpScsrgeam2_bufferSizeExt(
      cusparse_handle, nrows_A, ncols_A, (void *)(&alpha), descrA, nnz_A,
      dvalues_A, drow_offsets_A, dcol_indices_A, (void *)&beta, descrB, nnz_B,
      dvalues_B, drow_offsets_B, dcol_indices_B, descrC, dvalues_C,
      drow_offsets_C, dcol_indices_C, &BufferSizeInBytes);

  if (BufferSizeInBytes > 0)
    CUDADriver::get_instance().malloc((void **)(&buffer), BufferSizeInBytes);

  // Determine drow_offsets_C and the total number of nonzero elements.
  CUSPARSEDriver::get_instance().cpXcsrgeam2Nnz(
      cusparse_handle, nrows_A, ncols_A, descrA, nnz_A, drow_offsets_A,
      dcol_indices_A, descrB, nnz_B, drow_offsets_B, dcol_indices_B, descrC,
      drow_offsets_C, nnzTotalDevHostPtr, buffer);

  int baseC;
  if (nullptr != nnzTotalDevHostPtr) {
    nnzC = *nnzTotalDevHostPtr;
  } else {
    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)(&nnzC), (void *)(drow_offsets_C + nrows_A), sizeof(int));
    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)(&baseC), (void *)(drow_offsets_C), sizeof(int));
    nnzC -= baseC;
  }

  CUDADriver::get_instance().malloc((void **)&dcol_indices_C,
                                    sizeof(int) * nnzC);
  CUDADriver::get_instance().malloc((void **)&dvalues_C, sizeof(float) * nnzC);

  CUSPARSEDriver::get_instance().cpScsrgeam2(
      cusparse_handle, nrows_A, ncols_A, (void *)(&alpha), descrA, nnz_A,
      dvalues_A, drow_offsets_A, dcol_indices_A, (void *)(&beta), descrB, nnz_B,
      dvalues_B, drow_offsets_B, dcol_indices_B, descrC, dvalues_C,
      drow_offsets_C, dcol_indices_C, buffer);

  cusparseSpMatDescr_t matrix_C;
  CUSPARSEDriver::get_instance().cpCreateCsr(
      &matrix_C, rows_, cols_, nnzC, drow_offsets_C, dcol_indices_C, dvalues_C,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F);

  CUSPARSEDriver::get_instance().cpDestroy(cusparse_handle);
  CUSPARSEDriver::get_instance().cpDestroyMatDescr(descrA);
  CUSPARSEDriver::get_instance().cpDestroyMatDescr(descrB);
  CUSPARSEDriver::get_instance().cpDestroyMatDescr(descrC);
  CUDADriver::get_instance().mem_free(buffer);
  return make_cu_sparse_matrix(matrix_C, rows_, cols_, PrimitiveType::f32);
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

std::unique_ptr<SparseMatrix> CuSparseMatrix::matmul(
    const CuSparseMatrix &other) const {
#if defined(TI_WITH_CUDA)
  return gemm(other, 1.0f, 1.0f);
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

// Reference:
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spgemm
std::unique_ptr<SparseMatrix> CuSparseMatrix::gemm(const CuSparseMatrix &other,
                                                   const float alpha,
                                                   const float beta) const {
#if defined(TI_WITH_CUDA)
  cusparseHandle_t handle;
  CUSPARSEDriver::get_instance().cpCreate(&handle);
  cusparseOperation_t op_A = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t op_B = CUSPARSE_OPERATION_NON_TRANSPOSE;

  size_t nrows_A = rows_;
  size_t ncols_B = other.cols_;
  auto mat_A = matrix_;
  auto mat_B = other.matrix_;

  // 1. create resulting matrix `C`
  cusparseSpMatDescr_t mat_C;
  CUSPARSEDriver::get_instance().cpCreateCsr(
      &mat_C, nrows_A, ncols_B, 0, nullptr, nullptr, nullptr,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F);

  // 2. create gemm descr
  cusparseSpGEMMDescr_t spgemm_desc;
  CUSPARSEDriver::get_instance().cpCreateSpGEMM(&spgemm_desc);

  // 3. ask buffer_size1 bytes for external memory
  void *d_buffer1;
  size_t buffer_size1 = 0;
  CUSPARSEDriver::get_instance().cpSpGEMM_workEstimation(
      handle, op_A, op_B, &alpha, this->matrix_, other.matrix_, &beta, mat_C,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size1, nullptr);
  CUDADriver::get_instance().malloc((void **)&d_buffer1, buffer_size1);
  // 4. inspect the matrices A and B to understand the memory requirement for
  // the next step
  CUSPARSEDriver::get_instance().cpSpGEMM_workEstimation(
      handle, op_A, op_B, &alpha, this->matrix_, other.matrix_, &beta, mat_C,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size1,
      d_buffer1);

  // 5. ask buffer_size2 bytes for external memory
  size_t buffer_size2 = 0;
  CUSPARSEDriver::get_instance().cpSpGEMM_compute(
      handle, op_A, op_B, &alpha, mat_A, mat_B, &beta, mat_C, CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size2, nullptr);
  void *d_buffer2;
  CUDADriver::get_instance().malloc((void **)&d_buffer2, buffer_size2);

  // 6. compute the intermediate product of A * B
  CUSPARSEDriver::get_instance().cpSpGEMM_compute(
      handle, op_A, op_B, &alpha, mat_A, mat_B, &beta, mat_C, CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size2, d_buffer2);

  // 7. get info of matrix C
  size_t nrows_C, cols_C, nnz_C;
  CUSPARSEDriver::get_instance().cpGetSize(mat_C, &nrows_C, &cols_C, &nnz_C);

  // 8. allocate matric C
  int *d_csr_row_ptr_C, *d_csr_col_ind_C;
  float *d_values_C;
  CUDADriver::get_instance().malloc((void **)&d_csr_row_ptr_C,
                                    (nrows_A + 1) * sizeof(int));
  CUDADriver::get_instance().malloc((void **)&d_csr_col_ind_C,
                                    nnz_C * sizeof(int));
  CUDADriver::get_instance().malloc((void **)&d_values_C,
                                    nnz_C * sizeof(float));

  // 9. update matrix C with new pointers
  CUSPARSEDriver::get_instance().cpCsrSetPointers(mat_C, d_csr_row_ptr_C,
                                                  d_csr_col_ind_C, d_values_C);

  // 10. copy the final products of C.
  CUSPARSEDriver::get_instance().cpSpGEMM_copy(
      handle, op_A, op_B, &alpha, mat_A, mat_B, &beta, mat_C, CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, spgemm_desc);

  CUDADriver::get_instance().mem_free(d_buffer1);
  CUDADriver::get_instance().mem_free(d_buffer2);
  CUSPARSEDriver::get_instance().cpDestroy(handle);
  CUSPARSEDriver::get_instance().cpDestroySpGEMM(spgemm_desc);

  return make_cu_sparse_matrix(mat_C, nrows_A, ncols_B, PrimitiveType::f32);
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

// Convert CSR to CSC format using routine `Csr2cscEx2`
// to implement transpose.
// Reference
// https://stackoverflow.com/questions/57368010/how-to-transpose-a-sparse-matrix-in-cusparse
std::unique_ptr<SparseMatrix> CuSparseMatrix::transpose() const {
#if defined(TI_WITH_CUDA)
  cusparseHandle_t handle;
  CUSPARSEDriver::get_instance().cpCreate(&handle);
  size_t nrows_A, ncols_A, nnz;
  void *d_csr_val = nullptr, *d_csr_val_AT = nullptr;
  int *d_csr_row_ptr = nullptr, *d_csr_col_ind = nullptr;
  int *d_csr_row_ptr_AT = nullptr, *d_csr_col_ptr_AT = nullptr;
  cusparseIndexType_t csr_row_otr_type, csr_col_otr_type;
  cusparseIndexBase_t idx_base_type;
  cudaDataType value_type;
  size_t buffer_size;

  // 1. get pointers of A
  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &nrows_A, &ncols_A, &nnz, (void **)&d_csr_row_ptr,
      (void **)&d_csr_col_ind, (void **)&d_csr_val, &csr_row_otr_type,
      &csr_col_otr_type, &idx_base_type, &value_type);

  // 2. ask bufer size for Csr2cscEx2
  CUSPARSEDriver::get_instance().cpCsr2cscEx2_bufferSize(
      handle, nrows_A, ncols_A, nnz, (void *)&d_csr_val, (int *)&d_csr_row_ptr,
      (int *)&d_csr_col_ind, (void *)&d_csr_val_AT, (int *)&d_csr_row_ptr_AT,
      (int *)&d_csr_col_ptr_AT, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffer_size);
  void *buffer = nullptr;
  CUDADriver::get_instance().malloc((void **)&buffer, buffer_size);

  CUDADriver::get_instance().malloc((void **)&d_csr_val_AT,
                                    nnz * sizeof(float));
  CUDADriver::get_instance().malloc((void **)&d_csr_row_ptr_AT,
                                    (ncols_A + 1) * sizeof(int));
  CUDADriver::get_instance().malloc((void **)&d_csr_col_ptr_AT,
                                    nnz * sizeof(int));

  // 3. execute Csr2cscEx2
  CUSPARSEDriver::get_instance().cpCsr2cscEx2(
      handle, nrows_A, ncols_A, nnz, d_csr_val, d_csr_row_ptr, d_csr_col_ind,
      d_csr_val_AT, d_csr_row_ptr_AT, d_csr_col_ptr_AT, CUDA_R_32F,
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
      buffer);

  // 4. create AT.
  cusparseSpMatDescr_t mat_AT;
  CUSPARSEDriver::get_instance().cpCreateCsr(
      &mat_AT, ncols_A, nrows_A, nnz, (void *)d_csr_row_ptr_AT,
      (void *)d_csr_col_ptr_AT, (void *)d_csr_val_AT, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CUDADriver::get_instance().mem_free(buffer);
  CUSPARSEDriver::get_instance().cpDestroy(handle);
  return make_cu_sparse_matrix(mat_AT, ncols_A, nrows_A, PrimitiveType::f32);
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

void CuSparseMatrix::spmv(Program *prog, const Ndarray &x, Ndarray &y) {
#if defined(TI_WITH_CUDA)
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t dY = prog->get_ndarray_data_ptr_as_int(&y);

  cusparseDnVecDescr_t vecX, vecY;
  CUSPARSEDriver::get_instance().cpCreateDnVec(&vecX, cols_, (void *)dX,
                                               CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpCreateDnVec(&vecY, rows_, (void *)dY,
                                               CUDA_R_32F);

  cusparseHandle_t cusparse_handle;
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handle);
  float alpha = 1.0f, beta = 0.0f;
  size_t bufferSize = 0;
  CUSPARSEDriver::get_instance().cpSpMV_bufferSize(
      cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_, vecX,
      &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &bufferSize);

  void *dBuffer = nullptr;
  if (bufferSize > 0)
    CUDADriver::get_instance().malloc(&dBuffer, bufferSize);
  CUSPARSEDriver::get_instance().cpSpMV(
      cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_, vecX,
      &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, dBuffer);

  CUSPARSEDriver::get_instance().cpDestroyDnVec(vecX);
  CUSPARSEDriver::get_instance().cpDestroyDnVec(vecY);
  CUSPARSEDriver::get_instance().cpDestroy(cusparse_handle);
  CUDADriver::get_instance().mem_free(dBuffer);
#endif
}

const std::string CuSparseMatrix::to_string() const {
  std::ostringstream ostr;
#ifdef TI_WITH_CUDA
  size_t rows, cols, nnz;
  float *dR;
  int *dC, *dV;
  cusparseIndexType_t row_type, column_type;
  cusparseIndexBase_t idx_base;
  cudaDataType value_type;
  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &rows, &cols, &nnz, (void **)&dR, (void **)&dC, (void **)&dV,
      &row_type, &column_type, &idx_base, &value_type);

  auto *hR = new int[rows + 1];
  auto *hC = new int[nnz];
  auto *hV = new float[nnz];

  CUDADriver::get_instance().memcpy_device_to_host((void *)hR, (void *)dR,
                                                   (rows + 1) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hC, (void *)dC,
                                                   (nnz) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hV, (void *)dV,
                                                   (nnz) * sizeof(float));

  print_triplet_from_csr<int, int, float>(rows, cols, hR, hC, hV, ostr);
#endif
  return ostr.str();
}

}  // namespace taichi::lang
