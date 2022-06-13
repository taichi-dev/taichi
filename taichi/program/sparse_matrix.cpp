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

}  // namespace lang
}  // namespace taichi
