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
}  // namespace

namespace taichi {
namespace lang {

SparseMatrixBuilder::SparseMatrixBuilder(int rows,
                                         int cols,
                                         int max_num_triplets,
                                         DataType dtype,
                                         const std::string &storage_format)
    : rows_(rows),
      cols_(cols),
      max_num_triplets_(max_num_triplets),
      dtype_(dtype),
      storage_format_(storage_format) {
  auto element_size = data_type_size(dtype);
  TI_ASSERT((element_size == 4 || element_size == 8));
  data_base_ptr_ =
      std::make_unique<uchar[]>(max_num_triplets_ * 3 * element_size);
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
  auto element_size = data_type_size(dtype_);
  switch (element_size) {
    case 4:
      print_template<float32, int32>();
      break;
    case 8:
      print_template<float64, int64>();
      break;
    default:
      TI_ERROR("Unsupported sparse matrix data type!");
      break;
  }
}

template <typename T, typename G>
void SparseMatrixBuilder::build_template(std::unique_ptr<SparseMatrix> &m) {
  using V = Eigen::Triplet<T>;
  std::vector<V> triplets;
  T *data = reinterpret_cast<T *>(data_base_ptr_.get());
  for (int i = 0; i < num_triplets_; i++) {
    triplets.push_back(V(((G *)data)[i * 3], ((G *)data)[i * 3 + 1],
                         taichi_union_cast<T>(data[i * 3 + 2])));
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
  void *csr_row_offset_ptr = NULL;
  CUDADriver::get_instance().malloc(&csr_row_offset_ptr,
                                    sizeof(int) * (rows_ + 1));
  cusparseHandle_t cusparse_handle;
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handle);
  CUSPARSEDriver::get_instance().cpCoo2Csr(
      cusparse_handle, (void *)coo_row_ptr, nnz, rows_,
      (void *)csr_row_offset_ptr, CUSPARSE_INDEX_BASE_ZERO);

  CUSPARSEDriver::get_instance().cpCreateCsr(
      &matrix_, rows_, cols_, nnz, csr_row_offset_ptr, coo_col_ptr,
      coo_values_ptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpDestroy(cusparse_handle);
  // TODO: not sure if this array should be deleted now.
  // CUDADriver::get_instance().mem_free(csr_row_offset_ptr);
#endif
}

CuSparseMatrix::~CuSparseMatrix() {
#if defined(TI_WITH_CUDA)
  // CUSPARSEDriver::get_instance().cpDestroySpMat(matrix_);
#endif
}
void make_sparse_matrix_from_ndarray_cusparse(Program *prog,
                                              SparseMatrix &sm,
                                              const Ndarray &row_indices,
                                              const Ndarray &col_indices,
                                              const Ndarray &values) {
#if defined(TI_WITH_CUDA)
  std::string sdtype = taichi::lang::data_type_name(sm.get_data_type());
  if (!CUSPARSEDriver::get_instance().is_loaded()) {
    bool load_success = CUSPARSEDriver::get_instance().load_cusparse();
    if (!load_success) {
      TI_ERROR("Failed to load cusparse library!");
    }
  }
  size_t row_coo = prog->get_ndarray_data_ptr_as_int(&row_indices);
  size_t col_coo = prog->get_ndarray_data_ptr_as_int(&col_indices);
  size_t values_coo = prog->get_ndarray_data_ptr_as_int(&values);
  int nnz = values.get_nelement();
  sm.build_csr_from_coo((void *)row_coo, (void *)col_coo, (void *)values_coo,
                        nnz);
#endif
}

// Reference::https://docs.nvidia.com/cuda/cusparse/index.html#csrgeam2
const CuSparseMatrix CuSparseMatrix::addition(const CuSparseMatrix &other,
                                              const float alpha,
                                              const float beta) const {
#if defined(TI_WITH_CUDA)
  // Get information of this matrix: A
  size_t nrows_A = 0, ncols_A = 0, nnz_A = 0;
  void *drow_offsets_A = NULL, *dcol_indices_A = NULL, *dvalues_A = NULL;
  cusparseIndexType_t csrRowOffsetsType_A, csrColIndType_A;
  cusparseIndexBase_t idxBase_A;
  cudaDataType valueType_A;
  TI_ASSERT(matrix_ != NULL);

  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &nrows_A, &ncols_A, &nnz_A, &drow_offsets_A, &dcol_indices_A,
      &dvalues_A, &csrRowOffsetsType_A, &csrColIndType_A, &idxBase_A,
      &valueType_A);
  // Get information of other matrix: B
  size_t nrows_B = 0, ncols_B = 0, nnz_B = 0;
  void *drow_offsets_B = NULL, *dcol_indices_B = NULL, *dvalues_B = NULL;
  cusparseIndexType_t csrRowOffsetsType_B, csrColIndType_B;
  cusparseIndexBase_t idxBase_B;
  cudaDataType valueType_B;
  CUSPARSEDriver::get_instance().cpCsrGet(
      other.matrix_, &nrows_B, &ncols_B, &nnz_B, &drow_offsets_B,
      &dcol_indices_B, &dvalues_B, &csrRowOffsetsType_B, &csrColIndType_B,
      &idxBase_B, &valueType_B);

  // Create sparse matrix: C
  int *drow_offsets_C = NULL;
  int *dcol_indices_C = NULL;
  float *dvalues_C = NULL;
  cusparseMatDescr_t descrA = NULL, descrB = NULL, descrC = NULL;
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrA);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrB);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrC);
  CUSPARSEDriver::get_instance().cpSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);

  // Start to do addition
  cusparseHandle_t cusparse_handle;
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handle);
  // alpha, nnzTotalDevHostPtr points to host memory
  size_t BufferSizeInBytes;
  char *buffer = NULL;
  int nnzC;
  int *nnzTotalDevHostPtr = &nnzC;
  CUSPARSEDriver::get_instance().cpSetPointerMode(cusparse_handle,
                                                  CUSPARSE_POINTER_MODE_HOST);
  CUDADriver::get_instance().malloc((void**)(&drow_offsets_C), sizeof(int) * (nrows_A + 1));
  // Prepare buffer
  CUSPARSEDriver::get_instance().cpScsrgeam2_bufferSizeExt(
      cusparse_handle, nrows_A, ncols_A,
      (void*)(&alpha), 
      descrA, nnz_A,
      dvalues_A, drow_offsets_A, dcol_indices_A, 
      (void*)&beta, 
      descrB, nnz_B,
      dvalues_B, drow_offsets_B, dcol_indices_B, 
      descrC, 
      dvalues_C, drow_offsets_C, dcol_indices_C, 
      &BufferSizeInBytes);

  if (BufferSizeInBytes > 0)
    CUDADriver::get_instance().malloc((void**)(&buffer), BufferSizeInBytes);
    
  // Determine drow_offsets_C and the total number of nonzero elements.
  CUSPARSEDriver::get_instance().cpXcsrgeam2Nnz(
      cusparse_handle, nrows_A, ncols_A, 
      descrA, nnz_A, drow_offsets_A, dcol_indices_A, 
      descrB, nnz_B, drow_offsets_B, dcol_indices_B, 
      descrC, drow_offsets_C, nnzTotalDevHostPtr, buffer);

  int baseC;
  if (NULL != nnzTotalDevHostPtr) {
    nnzC = *nnzTotalDevHostPtr;
  } else {
    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)(&nnzC), (void *)(drow_offsets_C + nrows_A), sizeof(int));
    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)(&baseC), (void *)(drow_offsets_C), sizeof(int));
    nnzC -= baseC;
  }

  CUDADriver::get_instance().malloc((void **)&dcol_indices_C, sizeof(int) *nnzC); 
  CUDADriver::get_instance().malloc((void **)&dvalues_C, sizeof(float) * nnzC);

  CUSPARSEDriver::get_instance().cpScsrgeam2(
      cusparse_handle, nrows_A, ncols_A, 
      (void*)(&alpha), 
      descrA, nnz_A,
      dvalues_A, drow_offsets_A, dcol_indices_A, 
      (void*)(&beta), 
      descrB,
      nnz_B, 
      dvalues_B, drow_offsets_B, dcol_indices_B, 
      descrC, 
      dvalues_C, drow_offsets_C, dcol_indices_C, buffer);

  cusparseSpMatDescr_t matrix_C;
  CUSPARSEDriver::get_instance().cpCreateCsr(
      &matrix_C, rows_, cols_, nnzC,
      drow_offsets_C, dcol_indices_C, dvalues_C,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  CUSPARSEDriver::get_instance().cpDestroy(cusparse_handle);
  CUDADriver::get_instance().mem_free(buffer);
  return CuSparseMatrix(matrix_C, rows_, cols_, PrimitiveType::f32);
#endif
}

const CuSparseMatrix CuSparseMatrix::matmul(const CuSparseMatrix &other) const {
  return gemm(other, 1.0f, 1.0f);
}

// Reference: https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spgemm
const CuSparseMatrix CuSparseMatrix::gemm(const CuSparseMatrix &other,
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
  CUSPARSEDriver::get_instance().cpCreateCsr(&mat_C, nrows_A, ncols_B, 0,
                                              NULL, NULL, NULL,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // 2. create gemm descr 
  cusparseSpGEMMDescr_t spgemm_desc;
  CUSPARSEDriver::get_instance().cpSpCreateSpGEMM(&spgemm_desc);

  // 3. ask buffer_size1 bytes for external memory
  void * d_buffer1;
  size_t buffer_size1 = 0;
  CUSPARSEDriver::get_instance().cpSpGEMM_workEstimation(handle, op_A, op_B,
                                &alpha, this->matrix_, other.matrix_, &beta, mat_C,
                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                spgemm_desc, &buffer_size1, NULL);
  CUDADriver::get_instance().malloc((void**)& d_buffer1, buffer_size1);
  // 4. inspect the matrices A and B to understand the memory requirement for the next step
  CUSPARSEDriver::get_instance().cpSpGEMM_workEstimation(handle, op_A, op_B,
                                    &alpha, this->matrix_, other.matrix_, &beta, mat_C,
                                    CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemm_desc, &buffer_size1, d_buffer1);

  // 5. ask buffer_size2 bytes for external memory
  size_t buffer_size2 = 0;
  CUSPARSEDriver::get_instance().cpSpGEMM_compute(handle, op_A, op_B,
                                &alpha, mat_A, mat_B, &beta, mat_C,
                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                spgemm_desc, &buffer_size2, NULL);
  void *d_buffer2;
  CUDADriver::get_instance().malloc((void**)& d_buffer2, buffer_size2);

  // 6. compute the intermediate product of A * B
  CUSPARSEDriver::get_instance().cpSpGEMM_compute(handle, op_A, op_B,
                                &alpha, mat_A, mat_B, &beta, mat_C,
                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                spgemm_desc, &buffer_size2, d_buffer2);

  // 7. get info of matrix C 
  size_t nrows_C, cols_C, nnz_C;
  CUSPARSEDriver::get_instance().cpGetSize(mat_C, &nrows_C, &cols_C, &nnz_C);

  // 8. allocate matric C
  int *d_csr_row_ptr_C, *d_csr_col_ind_C;
  float* d_values_C;
  CUDADriver::get_instance().malloc((void**)&d_csr_row_ptr_C, (nrows_A+1) * sizeof(int));
  CUDADriver::get_instance().malloc((void**)&d_csr_col_ind_C, nnz_C * sizeof(int));
  CUDADriver::get_instance().malloc((void**)&d_values_C, nnz_C * sizeof(float));

  // 9. update matrix C with new pointers
  CUSPARSEDriver::get_instance().cpCsrSetPointers(mat_C, d_csr_row_ptr_C, d_csr_col_ind_C, d_values_C);

  // 10. copy the final products of C.
  CUSPARSEDriver::get_instance().cpSpGEMM_copy(handle, op_A, op_B,
                                &alpha, mat_A, mat_B, &beta, mat_C,
                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc);

  CUDADriver::get_instance().mem_free(d_buffer1);
  CUDADriver::get_instance().mem_free(d_buffer2);
  
  return CuSparseMatrix(mat_C, nrows_A, ncols_B, PrimitiveType::f32);
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

  void *dBuffer = NULL;
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
  print_helper();
  return ostr.str();
#if defined(TI_WITH_CUDA)
  size_t nrows_A = 0, ncols_A = 0, nnz_A = 0;
  void *drow_offsets_A = NULL, *dcol_indices_A = NULL, *dvalues_A = NULL;
  cusparseIndexType_t csrRowOffsetsType_A, csrColIndType_A;
  cusparseIndexBase_t idxBase_A;
  cudaDataType valueType_A;
  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &nrows_A, &ncols_A, &nnz_A, &drow_offsets_A, &dcol_indices_A,
      &dvalues_A, &csrRowOffsetsType_A, &csrColIndType_A, &idxBase_A,
      &valueType_A);

  int *h_row_offsets = (int *)malloc(sizeof(int) * (nrows_A + 1));
  int *h_col_indices = (int *)malloc(sizeof(int) * nnz_A);
  float *h_values = (float *)malloc(sizeof(float) * nnz_A);

  assert(h_row_offsets != NULL);
  assert(h_col_indices != NULL);
  assert(h_values != NULL);

  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)h_col_indices, dcol_indices_A, sizeof(int) * nnz_A);
  CUDADriver::get_instance().memcpy_device_to_host((void *)h_values, dvalues_A,
                                                   sizeof(float) * nnz_A);
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)h_row_offsets, drow_offsets_A, sizeof(int) * (nrows_A + 1));

  ostr << "nrows_A: " << nrows_A << ", ncols_A: " << ncols_A
       << ", nnz_A: " << nnz_A << ".\n";

  ostr << "row_offsets: "
       << "\n";
  for (int i = 0; i < nrows_A + 1; i++)
    ostr << h_row_offsets[i] << "\t";
  ostr << "\n";
  ostr << "col_indices: "
       << "\n";
  for (int i = 0; i < nnz_A; i++)
    ostr << h_col_indices[i] << "\t";
  ostr << "\n";
  ostr << "values: "
       << "\n";
  for (int i = 0; i < nnz_A; i++)
    ostr << h_values[i] << "\t";
  ostr << "\n";

  if (h_row_offsets)
    free(h_row_offsets);
  if (h_col_indices)
    free(h_col_indices);
  if (h_values)
    free(h_values);
#endif
  return ostr.str();
}

template <typename T, typename T1, typename T2>
void csr_to_triplet(int64_t n_rows, int n_cols, T* row, T1* col, T2* value) {
  using Triplets = Eigen::Triplet<T2>;
  std::vector<Triplets> trips;
  for (int64_t i = 1; i <= n_rows; ++i) {
      auto n_i = row[i] - row[i - 1];
      for (auto j = 0; j < n_i; ++j) {
        trips.push_back({i-1,col[row[i-1]+j],value[row[i-1]+j]});
      }
  }
  Eigen::SparseMatrix<float> m(n_rows, n_cols);
  m.setFromTriplets(trips.begin(), trips.end());
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  std::cout << Eigen::MatrixXf(m.cast<float>()).format(clean_fmt) << std::endl;
}

void CuSparseMatrix::print_helper() const {
#if defined(TI_WITH_CUDA)
  size_t rows, cols, nnz;
  float* dR, *dC, *dV;
  cusparseIndexType_t row_type, column_type;
  cusparseIndexBase_t idx_base;
  cudaDataType value_type;
  CUSPARSEDriver::get_instance().cpCsrGet(matrix_, &rows, &cols, &nnz, (void**)&dR, (void**)&dC, (void**)&dV, 
                    &row_type, &column_type, &idx_base, &value_type);
  
  auto* hR = new int[rows+1];
  auto* hC = new int[nnz];
  auto* hV = new float[nnz];

  CUDADriver::get_instance().memcpy_device_to_host(
        (void *)hR, (void *)dR, (rows+1) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host(
        (void *)hC, (void *)dC, (nnz) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host(
        (void *)hV, (void *)dV, (nnz) * sizeof(float));

  // std::cout << (row_type == CUSPARSE_INDEX_32I) << '\n';
  // std::cout << (column_type == CUSPARSE_INDEX_32I) << '\n';
  // std::cout << (value_type == CUDA_R_32F) << '\n';

  csr_to_triplet<int, int, float>(rows, cols, hR, hC, hV);
  
#endif
}

// const std::string CuSparseMatrix::to_string() const {
//   std::ostringstream ostr;
//   print_helper();
//   return ostr.str();
// }

}  // namespace lang
}  // namespace taichi
