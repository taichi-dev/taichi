#pragma once

#include "taichi/common/core.h"
#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/rhi/cuda/cuda_driver.h"

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
  std::string storage_format_{"col_major"};
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

  virtual void build_triplets(void *triplets_adr) {
    TI_NOT_IMPLEMENTED;
  };

  virtual void build_csr_from_coo(void *coo_row_ptr,
                                  void *coo_col_ptr,
                                  void *coo_values_ptr,
                                  int nnz) {
    TI_NOT_IMPLEMENTED;
  }
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

  inline DataType get_data_type() {
    return dtype_;
  }

  template <class T>
  T get_element(int row, int col) {
    std::cout << "get_element not implemented" << std::endl;
    return 0;
  }

  template <class T>
  void set_element(int row, int col, T value) {
    std::cout << "set_element not implemented" << std::endl;
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
  explicit EigenSparseMatrix(int rows, int cols, DataType dt)
      : SparseMatrix(rows, cols, dt), matrix_(rows, cols) {
  }
  explicit EigenSparseMatrix(EigenSparseMatrix &sm)
      : SparseMatrix(sm.num_rows(), sm.num_cols(), sm.dtype_),
        matrix_(sm.matrix_) {
  }
  explicit EigenSparseMatrix(EigenSparseMatrix &&sm)
      : SparseMatrix(sm.num_rows(), sm.num_cols(), sm.dtype_),
        matrix_(sm.matrix_) {
  }
  explicit EigenSparseMatrix(const EigenMatrix &em)
      : SparseMatrix(em.rows(), em.cols()), matrix_(em) {
  }

  ~EigenSparseMatrix() override = default;
  void build_triplets(void *triplets_adr) override;
  const std::string to_string() const override;

  const void *get_matrix() const override {
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

  friend EigenSparseMatrix operator*(const EigenSparseMatrix &lhs,
                                     const EigenSparseMatrix &rhs) {
    return EigenSparseMatrix(lhs.matrix_.cwiseProduct(rhs.matrix_));
  }

  EigenSparseMatrix transpose() {
    return EigenSparseMatrix(matrix_.transpose());
  }

  EigenSparseMatrix matmul(const EigenSparseMatrix &sm) {
    return EigenSparseMatrix(matrix_ * sm.matrix_);
  }

  template <typename T>
  T get_element(int row, int col) {
    return matrix_.coeff(row, col);
  }

  template <typename T>
  void set_element(int row, int col, T value) {
    matrix_.coeffRef(row, col) = value;
  }

  template <class VT>
  VT mat_vec_mul(const Eigen::Ref<const VT> &b) {
    return matrix_ * b;
  }

 private:
  EigenMatrix matrix_;
};

class CuSparseMatrix : public SparseMatrix {
 public:
  explicit CuSparseMatrix(int rows, int cols, DataType dt)
      : SparseMatrix(rows, cols, dt) {
#if defined(TI_WITH_CUDA)
    if (!CUSPARSEDriver::get_instance().is_loaded()) {
      bool load_success = CUSPARSEDriver::get_instance().load_cusparse();
      if (!load_success) {
        TI_ERROR("Failed to load cusparse library!");
      }
    }
#endif
  }

  virtual ~CuSparseMatrix();
  void build_csr_from_coo(void *coo_row_ptr,
                          void *coo_col_ptr,
                          void *coo_values_ptr,
                          int nnz) override;
  void spmv(Program *prog, const Ndarray &x, Ndarray &y);

  const void *get_matrix() const override {
    return &matrix_;
  };

  void print_info();

 private:
  cusparseSpMatDescr_t matrix_;
};

std::unique_ptr<SparseMatrix> make_sparse_matrix(
    int rows,
    int cols,
    DataType dt,
    const std::string &storage_format);
std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(int rows,
                                                    int cols,
                                                    DataType dt);

void make_sparse_matrix_from_ndarray(Program *prog,
                                     SparseMatrix &sm,
                                     const Ndarray &ndarray);
void make_sparse_matrix_from_ndarray_cusparse(Program *prog,
                                              SparseMatrix &sm,
                                              const Ndarray &row_indices,
                                              const Ndarray &col_indices,
                                              const Ndarray &values);
}  // namespace lang
}  // namespace taichi
