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
  SparseMatrixBuilder(int rows,
                      int cols,
                      int max_num_triplets,
                      DataType dtype,
                      const std::string &storage_format);

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
  std::string storage_format{"col_major"};
};

class SparseMatrix {
 public:
  SparseMatrix() : rows_(0), cols_(0), dtype_(PrimitiveType::f32){};
  SparseMatrix(int rows, int cols, DataType dt = PrimitiveType::f32)
      : rows_{rows}, cols_(cols), dtype_(dt){};
  SparseMatrix(SparseMatrix &sm)
      : rows_(sm.rows_), cols_(sm.cols_), dtype_(sm.dtype_) {
  }
  SparseMatrix(SparseMatrix &&sm)
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

  virtual float32 get_element(int row, int col) {
    return 0;
  }

  virtual void set_element(int row, int col, float32 value) {
    return;
  }

 protected:
  int rows_{0};
  int cols_{0};
  DataType dtype_{PrimitiveType::f32};
};

template <class EigenMatrix>
class EigenSparseMatrix : public SparseMatrix {
 public:
  EigenSparseMatrix(int rows, int cols, DataType dt)
      : SparseMatrix(rows, cols, dt), matrix_(rows, cols) {
  }
  explicit EigenSparseMatrix(EigenSparseMatrix &sm)
      : SparseMatrix(sm.num_rows(), sm.num_cols(), sm.dtype_),
        matrix_(sm.matrix_) {
  }
  EigenSparseMatrix(EigenSparseMatrix &&sm)
      : SparseMatrix(sm.num_rows(), sm.num_cols(), sm.dtype_),
        matrix_(sm.matrix_) {
  }
  explicit EigenSparseMatrix(const EigenMatrix &em)
      : SparseMatrix(em.rows(), em.cols()), matrix_(em) {
  }

  virtual ~EigenSparseMatrix() override = default;
  virtual void build_triplets(void *triplets_adr) override;
  virtual const std::string to_string() const override;

  virtual const void *get_matrix() const override {
    return &matrix_;
  };

  virtual EigenSparseMatrix &operator+=(const EigenSparseMatrix &other) {
    this->matrix_ += other.matrix_;
    return *this;
  };

  friend EigenSparseMatrix operator+(const EigenSparseMatrix &lhs,
                                     const EigenSparseMatrix &rhs) {
    return EigenSparseMatrix(lhs.matrix_ + rhs.matrix_);
  };

  virtual EigenSparseMatrix &operator-=(const EigenSparseMatrix &other) {
    this->matrix_ -= other.matrix_;
    return *this;
  }

  friend EigenSparseMatrix operator-(const EigenSparseMatrix &lhs,
                                     const EigenSparseMatrix &rhs) {
    return EigenSparseMatrix(lhs.matrix_ - rhs.matrix_);
  };

  virtual EigenSparseMatrix &operator*=(float scale) {
    this->matrix_ *= scale;
    return *this;
  }

  friend EigenSparseMatrix operator*(const EigenSparseMatrix &sm, float scale) {
    return EigenSparseMatrix(sm.matrix_ * scale);
  }

  friend EigenSparseMatrix operator*(float scale, const EigenSparseMatrix &sm) {
    return EigenSparseMatrix(sm.matrix_ * scale);
  }

  EigenSparseMatrix transpose() {
    return EigenSparseMatrix(matrix_.transpose());
  }

  EigenSparseMatrix matmul(const EigenSparseMatrix &sm) {
    return EigenSparseMatrix(matrix_ * sm.matrix_);
  }

  virtual float32 get_element(int row, int col) override {
    return matrix_.coeff(row, col);
  }

  void set_element(int row, int col, float32 value) override {
    matrix_.coeffRef(row, col) = value;
  }

  template <class VT>
  VT mat_vec_mul(const Eigen::Ref<const VT> &b) {
    return matrix_ * b;
  }

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
