#pragma once

#include "taichi/common/core.h"
#include "taichi/inc/constants.h"
#include "Eigen/Sparse"

namespace taichi {
namespace lang {

class SparseMatrix;

class SparseMatrixBuilder {
 public:
  SparseMatrixBuilder(int n, int m, int max_num_triplets);

  void *get_data_base_ptr();

  void print_triplets();

  SparseMatrix build();

 private:
  uint64 num_triplets_{0};
  void *data_base_ptr_{nullptr};
  std::vector<uint32> data_;
  int n_, m_;
  uint64 max_num_triplets_;
  bool built_{false};
};

class SparseMatrix {
 public:
  SparseMatrix() = delete;
  SparseMatrix(int n, int m);
  SparseMatrix(Eigen::SparseMatrix<float32> &matrix);

  int num_rows();
  int num_cols();
  void print();
  Eigen::SparseMatrix<float32> &get_matrix();
  float32 get_coeff(int row, int col);

  friend SparseMatrix operator+(const SparseMatrix &sm1,
                                const SparseMatrix &sm2);
  friend SparseMatrix operator-(const SparseMatrix &sm1,
                                const SparseMatrix &sm2);
  friend SparseMatrix operator*(float scale, const SparseMatrix &sm);
  friend SparseMatrix operator*(const SparseMatrix &sm1,
                                const SparseMatrix &sm2);
  SparseMatrix matmult(const SparseMatrix &sm);
  SparseMatrix transpose();

  void solve(SparseMatrix *);

 private:
  int n_, m_;
  Eigen::SparseMatrix<float32> matrix_;
};

}  // namespace lang
}  // namespace taichi
