#pragma once

#include "taichi/common/core.h"
#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"

#include "Eigen/Sparse"

namespace taichi {
namespace lang {

class SparseMatrix;

class SparseMatrixBuilder {
 public:
  SparseMatrixBuilder(int rows, int cols, int max_num_triplets, DataType dtype);

  ~SparseMatrixBuilder();
  void *get_data_base_ptr();

  void print_triplets();

  SparseMatrix build();

  void clear();

 private:
  template <typename T, typename G>
  void print_template();

  template <typename T, typename G>
  SparseMatrix build_template();

 private:
  uint64 num_triplets_{0};
  uchar *data_base_ptr_{nullptr};
  int rows_{0};
  int cols_{0};
  uint64 max_num_triplets_{0};
  bool built_{false};
  DataType dtype_{PrimitiveType::f32};
};

class SparseMatrix {
 public:
  SparseMatrix() = delete;
  SparseMatrix(int rows, int cols);
  SparseMatrix(Eigen::SparseMatrix<float32> &matrix);

  const int num_rows() const;
  const int num_cols() const;
  const std::string to_string() const;
  Eigen::SparseMatrix<float32> &get_matrix();
  const Eigen::SparseMatrix<float32> &get_matrix() const;
  float32 get_element(int row, int col);
  void set_element(int row, int col, float32 value);

  friend SparseMatrix operator+(const SparseMatrix &sm1,
                                const SparseMatrix &sm2);
  friend SparseMatrix operator-(const SparseMatrix &sm1,
                                const SparseMatrix &sm2);
  friend SparseMatrix operator*(float scale, const SparseMatrix &sm);
  friend SparseMatrix operator*(const SparseMatrix &sm, float scale);
  friend SparseMatrix operator*(const SparseMatrix &sm1,
                                const SparseMatrix &sm2);
  SparseMatrix matmul(const SparseMatrix &sm);
  Eigen::VectorXf mat_vec_mul(const Eigen::Ref<const Eigen::VectorXf> &b);

  SparseMatrix transpose();

 private:
  Eigen::SparseMatrix<float32, Eigen::ColMajor> matrix_;
};
}  // namespace lang
}  // namespace taichi
