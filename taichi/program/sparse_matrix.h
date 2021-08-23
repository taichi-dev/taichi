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

  void build(SparseMatrix* sm);


 private:
  uint64 num_triplets{0};
  void *data_base_ptr{nullptr};
  std::vector<uint32> data;
  int n, m;
  uint64 max_num_triplets;
  bool built{false};
};

class SparseMatrix{
public:
    SparseMatrix(){}
    SparseMatrix(int n, int m);
    SparseMatrix(Eigen::SparseMatrix<float32>& matrix);

    Eigen::SparseMatrix<float32>& get_matrix();
    void print();

    void solve(SparseMatrix* );

private:
  int n, m;
  Eigen::SparseMatrix<float32> matrix;
};


}  // namespace lang
}  // namespace taichi
