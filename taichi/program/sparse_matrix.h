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

  void print_triplets();

  std::unique_ptr<SparseMatrix> build();

  void clear();

 private:
  template <typename T, typename G>
  void print_template();

  template <typename T, typename G>
  void build_template(std::unique_ptr<SparseMatrix> &);

 private:
  uint64 num_triplets_{0};
  std::unique_ptr<uchar[]> data_base_ptr_{nullptr};
  int rows_{0};
  int cols_{0};
  uint64 max_num_triplets_{0};
  bool built_{false};
  DataType dtype_{PrimitiveType::f32};
};

class SparseMatrix {
 public:
  SparseMatrix() = delete;
  SparseMatrix(int rows, int cols, DataType dt = PrimitiveType::f32)
      : rows_{rows}, cols_(cols), dtype_(dt){};
  SparseMatrix(SparseMatrix &sm)
      : rows_(sm.rows_), cols_(sm.cols_), dtype_(sm.dtype_) {
  }
  virtual ~SparseMatrix() = default;

  virtual void build_triplets(void *triplets_adr){};

  inline const int num_rows() const {
    return rows_;
  }
  inline const int num_cols() const {
    return cols_;
  }
  virtual const std::string to_string() const {
    return nullptr;
  }
  virtual const void *get_matrix() const {
    return nullptr;
  }
  // Eigen::SparseMatrix<float32> &get_matrix();
  // const Eigen::SparseMatrix<float32> &get_matrix() const;
  // float32 get_element(int row, int col);
  // void set_element(int row, int col, float32 value);

  // friend SparseMatrix operator+(const SparseMatrix &sm1,
  //                               const SparseMatrix &sm2);
  // friend SparseMatrix operator-(const SparseMatrix &sm1,
  //                               const SparseMatrix &sm2);
  // friend SparseMatrix operator*(float scale, const SparseMatrix &sm);
  // friend SparseMatrix operator*(const SparseMatrix &sm, float scale);
  // friend SparseMatrix operator*(const SparseMatrix &sm1,
  //                               const SparseMatrix &sm2);
  // SparseMatrix matmul(const SparseMatrix &sm);
  // Eigen::VectorXf mat_vec_mul(const Eigen::Ref<const Eigen::VectorXf> &b);

  // SparseMatrix transpose();

 protected:
  int rows_{0};
  int cols_{0};
  DataType dtype_;
};

template <class EigenMatrix>
class EigenSparseMatrix : public SparseMatrix {
 public:
  EigenSparseMatrix(int rows, int cols, DataType dt)
      : SparseMatrix(rows, cols, dt), matrix_(rows, cols) {
  }
  EigenSparseMatrix(EigenSparseMatrix &sm)
      : SparseMatrix(sm.num_rows(), sm.num_cols(), sm.dtype_),
        matrix_(sm.matrix_) {
  }
  virtual ~EigenSparseMatrix() override = default;
  virtual void build_triplets(void *triplets_adr) override;
  virtual const std::string to_string() const override;

  virtual const void *get_matrix() const override {
    return &matrix_;
  };

 private:
  EigenMatrix matrix_;
};

std::unique_ptr<SparseMatrix> make_sparse_matrix(
    int rows,
    int cols,
    DataType dt,
    const std::string &storage_format);
}  // namespace lang
}  // namespace taichi
