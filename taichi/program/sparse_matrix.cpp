#include "taichi/program/sparse_matrix.h"

#include "Eigen/Dense"
#include "Eigen/SparseLU"

namespace taichi {
namespace lang {

SparseMatrixBuilder::SparseMatrixBuilder(int n, int m, int max_num_triplets)
    : n_(n), m_(m), max_num_triplets_(max_num_triplets) {
  data_.reserve(max_num_triplets * 3);
  data_base_ptr_ = get_data_base_ptr();
}

void *SparseMatrixBuilder::get_data_base_ptr() {
  return data_.data();
}

void SparseMatrixBuilder::print_triplets() {
  fmt::print("n={}, m={}, num_triplets={} (max={})", n_, m_, num_triplets_,
             max_num_triplets_);
  for (int64 i = 0; i < num_triplets_; i++) {
    fmt::print("({}, {}) val={}", data_[i * 3], data_[i * 3 + 1],
               taichi_union_cast<float32>(data_[i * 3 + 2]));
  }
  fmt::print("\n");
}

void SparseMatrixBuilder::build(SparseMatrix *sm) {
  TI_ASSERT(built == false);
  built_ = true;
  using T = Eigen::Triplet<float32>;
  std::vector<T> triplets;
  for (int i = 0; i < num_triplets_; i++) {
    triplets.push_back({data_[i * 3], data_[i * 3 + 1],
                        taichi_union_cast<float32>(data_[i * 3 + 2])});
  }
  sm->get_matrix().setFromTriplets(triplets.begin(), triplets.end());
}

SparseMatrix::SparseMatrix(Eigen::SparseMatrix<float32> &matrix) {
  this->matrix_ = matrix;
}

SparseMatrix::SparseMatrix(int n, int m) : n_(n), m_(m), matrix_(n, m) {
}

void SparseMatrix::print() {
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  // Note that the code below first converts the sparse matrix into a dense one.
  // https://stackoverflow.com/questions/38553335/how-can-i-print-in-console-a-formatted-sparse-matrix-with-eigen
  std::cout << Eigen::MatrixXf(matrix_).format(clean_fmt) << std::endl;
}

int SparseMatrix::num_rows() {
  return matrix_.rows();
}
int SparseMatrix::num_cols() {
  return matrix_.cols();
}

Eigen::SparseMatrix<float32> &SparseMatrix::get_matrix() {
  return matrix_;
}

SparseMatrix *operator+(const SparseMatrix &sm1, const SparseMatrix &sm2) {
  SparseMatrix *res = new SparseMatrix(sm1.n_, sm1.m_);
  res->matrix_ = sm1.matrix_ + sm2.matrix_;
  return res;
}

SparseMatrix *operator-(const SparseMatrix &sm1, const SparseMatrix &sm2) {
  SparseMatrix *res = new SparseMatrix(sm1.n_, sm1.m_);
  res->matrix_ = sm1.matrix_ - sm2.matrix_;
  return res;
}

SparseMatrix *operator*(float scale, const SparseMatrix &sm) {
  SparseMatrix *res = new SparseMatrix(sm.n_, sm.m_);
  res->matrix_ = scale * sm.matrix_;
  return res;
}

SparseMatrix *operator*(const SparseMatrix &sm1, const SparseMatrix &sm2) {
  SparseMatrix *res = new SparseMatrix(sm1.n_, sm1.m_);
  res->matrix_ = sm1.matrix_.cwiseProduct(sm2.matrix_);
  return res;
}

SparseMatrix *SparseMatrix::matmult(const SparseMatrix &sm) {
  SparseMatrix *res = new SparseMatrix(n_, m_);
  res->matrix_ = matrix_ * sm.matrix_;
  return res;
}

SparseMatrix *SparseMatrix::transpose() {
  SparseMatrix *res = new SparseMatrix(n_, m_);
  res->matrix_ = matrix_.transpose();
  return res;
}

float32 SparseMatrix::get_coeff(int row, int col) {
  return matrix_.coeff(row, col);
}

void SparseMatrix::solve(SparseMatrix *b_) {
  using namespace Eigen;

  VectorXf x(n_), b(m_);

  b.setZero();

  for (int k = 0; k < b_->matrix_.outerSize(); ++k)
    for (Eigen::SparseMatrix<float32>::InnerIterator it(b_->matrix_, k); it;
         ++it) {
      b[it.row()] = it.value();
    }

  x = Eigen::MatrixXf(matrix_).ldlt().solve(b);

  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  // Note that the code below first converts the sparse matrix into a dense one.
  // https://stackoverflow.com/questions/38553335/how-can-i-print-in-console-a-formatted-sparse-matrix-with-eigen
  std::cout << Eigen::MatrixXf(x).format(clean_fmt) << std::endl;
}

}  // namespace lang
}  // namespace taichi
