#include "taichi/program/sparse_matrix.h"

#include <sstream>

#include "Eigen/Dense"
#include "Eigen/SparseLU"

namespace taichi {
namespace lang {

SparseMatrixBuilder::SparseMatrixBuilder(int rows,
                                         int cols,
                                         int max_num_triplets,
                                         DataType dtype)
    : rows_(rows),
      cols_(cols),
      max_num_triplets_(max_num_triplets),
      dtype_(dtype) {
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
  auto sm = make_sparse_matrix(rows_, cols_, dtype_, "col");
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

// SparseMatrix::SparseMatrix(Eigen::SparseMatrix<float32> &matrix) {
//   this->matrix_ = matrix;
// }

// SparseMatrix::SparseMatrix(int rows, int cols) : matrix_(rows, cols) {
// }
template <class EigenMatrix>
const std::string EigenSparseMatrix<EigenMatrix>::to_string() const {
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  // Note that the code below first converts the sparse matrix into a dense one.
  // https://stackoverflow.com/questions/38553335/how-can-i-print-in-console-a-formatted-sparse-matrix-with-eigen
  std::ostringstream ostr;
  ostr << Eigen::MatrixXf(matrix_).format(clean_fmt);
  return ostr.str();
}

// const int SparseMatrix::num_rows() const {
//   return matrix_.rows();
// }
// const int SparseMatrix::num_cols() const {
//   return matrix_.cols();
// }

// Eigen::SparseMatrix<float32> &SparseMatrix::get_matrix() {
//   return matrix_;
// }

// const Eigen::SparseMatrix<float32> &SparseMatrix::get_matrix() const {
//   return matrix_;
// }

// SparseMatrix operator+(const SparseMatrix &sm1, const SparseMatrix &sm2) {
//   Eigen::SparseMatrix<float32> res(sm1.matrix_ + sm2.matrix_);
//   return SparseMatrix(res);
// }

// SparseMatrix operator-(const SparseMatrix &sm1, const SparseMatrix &sm2) {
//   Eigen::SparseMatrix<float32> res(sm1.matrix_ - sm2.matrix_);
//   return SparseMatrix(res);
// }

// SparseMatrix operator*(float scale, const SparseMatrix &sm) {
//   Eigen::SparseMatrix<float32> res(scale * sm.matrix_);
//   return SparseMatrix(res);
// }

// SparseMatrix operator*(const SparseMatrix &sm, float scale) {
//   return scale * sm;
// }

// SparseMatrix operator*(const SparseMatrix &sm1, const SparseMatrix &sm2) {
//   Eigen::SparseMatrix<float32> res(sm1.matrix_.cwiseProduct(sm2.matrix_));
//   return SparseMatrix(res);
// }

// SparseMatrix SparseMatrix::matmul(const SparseMatrix &sm) {
//   Eigen::SparseMatrix<float32> res(matrix_ * sm.matrix_);
//   return SparseMatrix(res);
// }

// Eigen::VectorXf SparseMatrix::mat_vec_mul(
//     const Eigen::Ref<const Eigen::VectorXf> &b) {
//   return matrix_ * b;
// }

// SparseMatrix SparseMatrix::transpose() {
//   Eigen::SparseMatrix<float32> res(matrix_.transpose());
//   return SparseMatrix(res);
// }

// float32 SparseMatrix::get_element(int row, int col) {
//   return matrix_.coeff(row, col);
// }

// void SparseMatrix::set_element(int row, int col, float32 value) {
//   matrix_.coeffRef(row, col) = value;
// }

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::build_triplets(void *triplets_adr) {
  if (taichi::lang::data_type_name(dtype_) == "f32") {
    using T = Eigen::Triplet<float32>;
    std::vector<T> *triplets = static_cast<std::vector<T> *>(triplets_adr);
    matrix_.setFromTriplets(triplets->begin(), triplets->end());
  }
}

std::unique_ptr<SparseMatrix> make_sparse_matrix(
    int rows,
    int cols,
    DataType dt,
    const std::string &storage_format) {
  if (taichi::lang::data_type_name(dt) == "f32" && storage_format == "cm") {
    using FC = Eigen::SparseMatrix<float, Eigen::ColMajor>;
    return std::make_unique<EigenSparseMatrix<FC>>(rows, cols, dt);
  }
}

}  // namespace lang
}  // namespace taichi
