#include "taichi/program/sparse_matrix.h"

#include <fstream>
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

#define INSTANTIATE_SPMV(type, storage)                               \
  template void                                                       \
  EigenSparseMatrix<Eigen::SparseMatrix<type, Eigen::storage>>::spmv( \
      Program *prog, const Ndarray &x, const Ndarray &y);

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
void print_triplets_from_csr(int64_t n_rows,
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

template <typename T, typename T1, typename T2>
T2 get_element_from_csr(int row,
                        int col,
                        T *row_data,
                        T1 *col_data,
                        T2 *value) {
  for (T i = row_data[row]; i < row_data[row + 1]; ++i) {
    if (col == col_data[i])
      return value[i];
  }
  // zero entry
  return 0;
}

}  // namespace

namespace taichi::lang {

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
}

SparseMatrixBuilder::~SparseMatrixBuilder() = default;

void SparseMatrixBuilder::create_ndarray(Program *prog) {
  ndarray_data_base_ptr_ = prog->create_ndarray(
      dtype_, std::vector<int>{3 * (int)max_num_triplets_ + 1});
  ndarray_data_ptr_ = prog->get_ndarray_data_ptr_as_int(ndarray_data_base_ptr_);
}

void SparseMatrixBuilder::delete_ndarray(Program *prog) {
  prog->delete_ndarray(ndarray_data_base_ptr_);
}

template <typename T, typename G>
void SparseMatrixBuilder::print_triplets_template() {
  auto ptr = get_ndarray_data_ptr();
  G *data = reinterpret_cast<G *>(ptr);
  num_triplets_ = data[0];
  fmt::print("n={}, m={}, num_triplets={} (max={})\n", rows_, cols_,
             num_triplets_, max_num_triplets_);
  data += 1;
  for (int i = 0; i < num_triplets_; i++) {
    fmt::print("[{}, {}] = {}\n", data[i * 3], data[i * 3 + 1],
               taichi_union_cast<T>(data[i * 3 + 2]));
  }
}

void SparseMatrixBuilder::print_triplets_eigen() {
  auto element_size = data_type_size(dtype_);
  switch (element_size) {
    case 4:
      print_triplets_template<float32, int32>();
      break;
    case 8:
      print_triplets_template<float64, int64>();
      break;
    default:
      TI_ERROR("Unsupported sparse matrix data type!");
      break;
  }
}

void SparseMatrixBuilder::print_triplets_cuda() {
#ifdef TI_WITH_CUDA
  CUDADriver::get_instance().memcpy_device_to_host(
      &num_triplets_, (void *)get_ndarray_data_ptr(), sizeof(int));
  fmt::print("n={}, m={}, num_triplets={} (max={})\n", rows_, cols_,
             num_triplets_, max_num_triplets_);
  auto len = 3 * num_triplets_ + 1;
  std::vector<float32> trips(len);
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)trips.data(), (void *)get_ndarray_data_ptr(),
      len * sizeof(float32));
  for (auto i = 0; i < num_triplets_; i++) {
    int row = taichi_union_cast<int>(trips[3 * i + 1]);
    int col = taichi_union_cast<int>(trips[3 * i + 2]);
    auto val = trips[i * 3 + 3];
    fmt::print("[{}, {}] = {}\n", row, col, val);
  }
#endif
}

intptr_t SparseMatrixBuilder::get_ndarray_data_ptr() const {
  return ndarray_data_ptr_;
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

std::unique_ptr<SparseMatrix> SparseMatrixBuilder::build_cuda() {
  TI_ASSERT(built_ == false);
  built_ = true;
  auto sm = make_cu_sparse_matrix(rows_, cols_, dtype_);
#ifdef TI_WITH_CUDA
  CUDADriver::get_instance().memcpy_device_to_host(
      &num_triplets_, (void *)get_ndarray_data_ptr(), sizeof(int));
  auto len = 3 * num_triplets_ + 1;
  std::vector<float32> trips(len);
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)trips.data(), (void *)get_ndarray_data_ptr(),
      len * sizeof(float32));
  std::unordered_map<int, std::tuple<int, int, float32>> entries;
  for (auto i = 0; i < num_triplets_; i++) {
    int row = taichi_union_cast<int>(trips[3 * i + 1]);
    int col = taichi_union_cast<int>(trips[3 * i + 2]);
    auto val = trips[i * 3 + 3];
    auto e_idx = row * cols_ + col;
    if (entries.find(e_idx) == entries.end()) {
      entries[e_idx] = std::make_tuple(row, col, val);
    } else {
      auto [r, c, v] = entries[e_idx];
      entries[e_idx] = std::make_tuple(r, c, v + val);
    }
  }
  auto entry_size = entries.size();
  int *row_host = (int *)malloc(sizeof(int) * entry_size);
  int *col_host = (int *)malloc(sizeof(int) * entry_size);
  float32 *value_host = (float32 *)malloc(sizeof(float32) * entry_size);
  int count = 0;
  for (auto entry : entries) {
    auto [row, col, value] = entry.second;
    row_host[count] = row;
    col_host[count] = col;
    value_host[count] = value;
    count++;
  }
  void *row_device = nullptr, *col_device = nullptr, *value_device = nullptr;
  CUDADriver::get_instance().malloc(&row_device, entry_size * sizeof(int));
  CUDADriver::get_instance().malloc(&col_device, entry_size * sizeof(int));
  CUDADriver::get_instance().malloc(&value_device,
                                    entry_size * sizeof(float32));
  CUDADriver::get_instance().memcpy_host_to_device(row_device, (void *)row_host,
                                                   entry_size * sizeof(int));
  CUDADriver::get_instance().memcpy_host_to_device(col_device, (void *)col_host,
                                                   entry_size * sizeof(int));
  CUDADriver::get_instance().memcpy_host_to_device(
      value_device, (void *)value_host, entry_size * sizeof(float32));
  sm->build_csr_from_coo(row_device, col_device, value_device, entry_size);
  clear();
  free(row_host);
  free(col_host);
  free(value_host);
#endif
  return sm;
}

void SparseMatrixBuilder::clear() {
  built_ = false;
  ndarray_data_base_ptr_->write_int(std::vector<int>{0}, 0);
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
void EigenSparseMatrix<EigenMatrix>::mmwrite(const std::string &filename) {
  std::ofstream file(filename);
  file << "%%MatrixMarket matrix coordinate real general\n%" << std::endl;
  file << matrix_.rows() << " " << matrix_.cols() << " " << matrix_.nonZeros()
       << std::endl;
  for (int k = 0; k < matrix_.outerSize(); ++k) {
    for (typename EigenMatrix::InnerIterator it(matrix_, k); it; ++it) {
      file << it.row() + 1 << " " << it.col() + 1 << " " << it.value()
           << std::endl;
    }
  }
  file.close();
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

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::spmv(Program *prog,
                                          const Ndarray &x,
                                          const Ndarray &y) {
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t dY = prog->get_ndarray_data_ptr_as_int(&y);
  std::string sdtype = taichi::lang::data_type_name(dtype_);
  if (sdtype == "f32") {
    Eigen::Map<Eigen::VectorXf>((float *)dY, cols_) =
        matrix_.template cast<float>() *
        Eigen::Map<Eigen::VectorXf>((float *)dX, cols_);
  } else if (sdtype == "f64") {
    Eigen::Map<Eigen::VectorXd>((double *)dY, cols_) =
        matrix_.template cast<double>() *
        Eigen::Map<Eigen::VectorXd>((double *)dX, cols_);
  } else {
    TI_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
}

INSTANTIATE_SPMV(float32, ColMajor)
INSTANTIATE_SPMV(float32, RowMajor)
INSTANTIATE_SPMV(float64, ColMajor)
INSTANTIATE_SPMV(float64, RowMajor)

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
                                                    DataType dt,
                                                    void *csr_row_ptr,
                                                    void *csr_col_ind,
                                                    void *csr_val_,
                                                    int nnz) {
  return std::unique_ptr<SparseMatrix>(std::make_unique<CuSparseMatrix>(
      mat, rows, cols, dt, csr_row_ptr, csr_col_ind, csr_val_, nnz));
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
  if (vec_permutation)
    CUSPARSEDriver::get_instance().cpDestroySpVec(vec_permutation);
  if (vec_values)
    CUSPARSEDriver::get_instance().cpDestroyDnVec(vec_values);
  if (cusparse_handle)
    CUSPARSEDriver::get_instance().cpDestroy(cusparse_handle);
  if (coo_row_ptr)
    CUDADriver::get_instance().mem_free(coo_row_ptr);
  if (d_values_sorted)
    CUDADriver::get_instance().mem_free(d_values_sorted);
  if (d_permutation)
    CUDADriver::get_instance().mem_free(d_permutation);
  if (dbuffer)
    CUDADriver::get_instance().mem_free(dbuffer);
  csr_row_ptr_ = csr_row_offset_ptr;
  csr_col_ind_ = coo_col_ptr;
  csr_val_ = coo_values_ptr;
  nnz_ = nnz;
#endif
}

CuSparseMatrix::~CuSparseMatrix() {
#if defined(TI_WITH_CUDA)
  if (matrix_)
    CUSPARSEDriver::get_instance().cpDestroySpMat(matrix_);
  if (csr_row_ptr_)
    CUDADriver::get_instance().mem_free(csr_row_ptr_);
  if (csr_col_ind_)
    CUDADriver::get_instance().mem_free(csr_col_ind_);
  if (csr_val_)
    CUDADriver::get_instance().mem_free(csr_val_);
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
  return make_cu_sparse_matrix(matrix_C, rows_, cols_, PrimitiveType::f32,
                               drow_offsets_C, dcol_indices_C, dvalues_C, nnzC);
  ;
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

std::unique_ptr<SparseMatrix> CuSparseMatrix::matmul(
    const CuSparseMatrix &other) const {
#if defined(TI_WITH_CUDA)
  return gemm(other, 1.0f, 0.0f);
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
  cusparseHandle_t handle = nullptr;
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
  void *d_buffer1 = nullptr;
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
  void *d_buffer2 = nullptr;
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

  return make_cu_sparse_matrix(mat_C, nrows_A, ncols_B, PrimitiveType::f32,
                               d_csr_row_ptr_C, d_csr_col_ind_C, d_values_C,
                               nnz_C);
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
  return make_cu_sparse_matrix(mat_AT, ncols_A, nrows_A, PrimitiveType::f32,
                               d_csr_row_ptr_AT, d_csr_col_ptr_AT, d_csr_val_AT,
                               nnz);
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

void CuSparseMatrix::spmv(size_t dX, size_t dY) {
#if defined(TI_WITH_CUDA)
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

void CuSparseMatrix::nd_spmv(Program *prog,
                             const Ndarray &x,
                             const Ndarray &y) {
#if defined(TI_WITH_CUDA)
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t dY = prog->get_ndarray_data_ptr_as_int(&y);
  spmv(dX, dY);
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

  print_triplets_from_csr<int, int, float>(rows, cols, hR, hC, hV, ostr);
  delete[] hR;
  delete[] hC;
  delete[] hV;
#endif
  return ostr.str();
}

float CuSparseMatrix::get_element(int row, int col) const {
  float res = 0.0f;
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

  TI_ASSERT(row < rows);
  TI_ASSERT(col < cols);

  auto *hR = new int[rows + 1];
  auto *hC = new int[nnz];
  auto *hV = new float[nnz];

  CUDADriver::get_instance().memcpy_device_to_host((void *)hR, (void *)dR,
                                                   (rows + 1) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hC, (void *)dC,
                                                   (nnz) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hV, (void *)dV,
                                                   (nnz) * sizeof(float));

  res = get_element_from_csr<int, int, float>(row, col, hR, hC, hV);

  delete[] hR;
  delete[] hC;
  delete[] hV;
#endif  // TI_WITH_CUDA
  return res;
}

void CuSparseMatrix::mmwrite(const std::string &filename) {
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

  std::ofstream file(filename);
  file << "%%MatrixMarket matrix coordinate real general\n%" << std::endl;
  file << rows << " " << cols << " " << nnz << std::endl;
  for (int r = 0; r < rows; r++) {
    for (int c = hR[r]; c < hR[r + 1]; c++) {
      file << r + 1 << " " << hC[c] + 1 << " " << hV[c] << std::endl;
    }
  }
  file.close();
  delete[] hR;
  delete[] hC;
  delete[] hV;
#endif
}

}  // namespace taichi::lang
