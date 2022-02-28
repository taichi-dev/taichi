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
  data_base_ptr_ = new uint32[max_num_triplets_ * 3 * element_size];
  if (data_base_ptr_ == nullptr) {
    TI_ERROR("Failed to allocate memory for sparse matrix builder");
  }
}

SparseMatrixBuilder::~SparseMatrixBuilder() {
  // TODO: why this genenerates an segment fault error?
  // delete [] data_base_ptr_;
}

void *SparseMatrixBuilder::get_data_base_ptr() {
  return data_base_ptr_;
}

void SparseMatrixBuilder::print_triplets() {
  fmt::print("n={}, m={}, num_triplets={} (max={})", rows_, cols_,
             num_triplets_, max_num_triplets_);
  for (int64 i = 0; i < num_triplets_; i++) {
    fmt::print("({}, {}) val={}", data_base_ptr_[i * 3],
               data_base_ptr_[i * 3 + 1],
               taichi_union_cast<float32>(data_base_ptr_[i * 3 + 2]));
  }
  fmt::print("\n");
}

SparseMatrix SparseMatrixBuilder::build() {
  TI_ASSERT(built_ == false);
  built_ = true;
  using T = Eigen::Triplet<float32>;
  std::vector<T> triplets;
  for (int i = 0; i < num_triplets_; i++) {
    triplets.push_back(
        T(data_base_ptr_[i * 3], data_base_ptr_[i * 3 + 1],
          taichi_union_cast<float32>(data_base_ptr_[i * 3 + 2])));
  }
  SparseMatrix sm(rows_, cols_);
  sm.get_matrix().setFromTriplets(triplets.begin(), triplets.end());
  clear();
  return sm;
}

void SparseMatrixBuilder::clear() {
  built_ = false;
  num_triplets_ = 0;
}

SparseMatrix::SparseMatrix(Eigen::SparseMatrix<float32> &matrix) {
  this->matrix_ = matrix;
}

SparseMatrix::SparseMatrix(int rows, int cols) : matrix_(rows, cols) {
}

const std::string SparseMatrix::to_string() const {
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  // Note that the code below first converts the sparse matrix into a dense one.
  // https://stackoverflow.com/questions/38553335/how-can-i-print-in-console-a-formatted-sparse-matrix-with-eigen
  std::ostringstream ostr;
  ostr << Eigen::MatrixXf(matrix_).format(clean_fmt);
  return ostr.str();
}

const int SparseMatrix::num_rows() const {
  return matrix_.rows();
}
const int SparseMatrix::num_cols() const {
  return matrix_.cols();
}

Eigen::SparseMatrix<float32> &SparseMatrix::get_matrix() {
  return matrix_;
}

const Eigen::SparseMatrix<float32> &SparseMatrix::get_matrix() const {
  return matrix_;
}

SparseMatrix operator+(const SparseMatrix &sm1, const SparseMatrix &sm2) {
  Eigen::SparseMatrix<float32> res(sm1.matrix_ + sm2.matrix_);
  return SparseMatrix(res);
}

SparseMatrix operator-(const SparseMatrix &sm1, const SparseMatrix &sm2) {
  Eigen::SparseMatrix<float32> res(sm1.matrix_ - sm2.matrix_);
  return SparseMatrix(res);
}

SparseMatrix operator*(float scale, const SparseMatrix &sm) {
  Eigen::SparseMatrix<float32> res(scale * sm.matrix_);
  return SparseMatrix(res);
}

SparseMatrix operator*(const SparseMatrix &sm, float scale) {
  return scale * sm;
}

SparseMatrix operator*(const SparseMatrix &sm1, const SparseMatrix &sm2) {
  Eigen::SparseMatrix<float32> res(sm1.matrix_.cwiseProduct(sm2.matrix_));
  return SparseMatrix(res);
}

SparseMatrix SparseMatrix::matmul(const SparseMatrix &sm) {
  Eigen::SparseMatrix<float32> res(matrix_ * sm.matrix_);
  return SparseMatrix(res);
}

Eigen::VectorXf SparseMatrix::mat_vec_mul(
    const Eigen::Ref<const Eigen::VectorXf> &b) {
  return matrix_ * b;
}

SparseMatrix SparseMatrix::transpose() {
  Eigen::SparseMatrix<float32> res(matrix_.transpose());
  return SparseMatrix(res);
}

float32 SparseMatrix::get_element(int row, int col) {
  return matrix_.coeff(row, col);
}

void SparseMatrix::set_element(int row, int col, float32 value) {
  matrix_.coeffRef(row, col) = value;
}

}  // namespace lang
}  // namespace taichi
