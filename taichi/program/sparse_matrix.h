#pragma once

#include "taichi/common/core.h"
#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/rhi/cuda/cuda_driver.h"

#include "Eigen/Sparse"

namespace taichi::lang {

class SparseMatrix;

class SparseMatrixBuilder {
 public:
  SparseMatrixBuilder(int rows,
                      int cols,
                      int max_num_triplets,
                      DataType dtype,
                      const std::string &storage_format);

  ~SparseMatrixBuilder();
  void print_triplets_eigen();
  void print_triplets_cuda();

  void create_ndarray(Program *prog);

  void delete_ndarray(Program *prog);

  intptr_t get_ndarray_data_ptr() const;

  std::unique_ptr<SparseMatrix> build();

  std::unique_ptr<SparseMatrix> build_cuda();

  void clear();

 private:
  template <typename T, typename G>
  void build_template(std::unique_ptr<SparseMatrix> &);

  template <typename T, typename G>
  void print_triplets_template();

 private:
  uint64 num_triplets_{0};
  Ndarray *ndarray_data_base_ptr_{nullptr};
  intptr_t ndarray_data_ptr_{0};
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
    return "";
  }

  virtual const void *get_matrix() const {
    return nullptr;
  }

  inline const DataType get_data_type() const {
    return dtype_;
  }

  template <class T>
  T get_element(int row, int col) {
    TI_NOT_IMPLEMENTED;
  }

  template <class T>
  void set_element(int row, int col, T value) {
    TI_NOT_IMPLEMENTED;
  }

  virtual void mmwrite(const std::string &filename) {
    TI_NOT_IMPLEMENTED;
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
  EigenSparseMatrix(EigenSparseMatrix &sm)
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

  ~EigenSparseMatrix() override = default;

  void build_triplets(void *triplets_adr) override;
  const std::string to_string() const override;

  // Write the sparse matrix to a Matrix Market file
  void mmwrite(const std::string &filename) override;

  const void *get_matrix() const override {
    return &matrix_;
  };

  void *get_matrix() {
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

  void spmv(Program *prog, const Ndarray &x, const Ndarray &y);

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
  explicit CuSparseMatrix(cusparseSpMatDescr_t A,
                          int rows,
                          int cols,
                          DataType dt,
                          void *csr_row_ptr,
                          void *csr_col_ind,
                          void *csr_val,
                          int nnz)
      : SparseMatrix(rows, cols, dt),
        matrix_(A),
        csr_row_ptr_(csr_row_ptr),
        csr_col_ind_(csr_col_ind),
        csr_val_(csr_val),
        nnz_(nnz) {
  }
  CuSparseMatrix(const CuSparseMatrix &sm)
      : SparseMatrix(sm.rows_, sm.cols_, sm.dtype_), matrix_(sm.matrix_) {
  }

  ~CuSparseMatrix() override;

  // TODO: Overload +=, -= and *=
  friend std::unique_ptr<SparseMatrix> operator+(const CuSparseMatrix &lhs,
                                                 const CuSparseMatrix &rhs) {
    auto m = lhs.addition(rhs, 1.0, 1.0);
    return m;
  };

  friend std::unique_ptr<SparseMatrix> operator-(const CuSparseMatrix &lhs,
                                                 const CuSparseMatrix &rhs) {
    return lhs.addition(rhs, 1.0, -1.0);
  };

  friend std::unique_ptr<SparseMatrix> operator*(const CuSparseMatrix &sm,
                                                 float scale) {
    return sm.addition(sm, scale, 0.0);
  }

  friend std::unique_ptr<SparseMatrix> operator*(float scale,
                                                 const CuSparseMatrix &sm) {
    return sm.addition(sm, scale, 0.0);
  }

  std::unique_ptr<SparseMatrix> addition(const CuSparseMatrix &other,
                                         const float alpha,
                                         const float beta) const;

  std::unique_ptr<SparseMatrix> matmul(const CuSparseMatrix &other) const;

  std::unique_ptr<SparseMatrix> gemm(const CuSparseMatrix &other,
                                     const float alpha,
                                     const float beta) const;

  std::unique_ptr<SparseMatrix> transpose() const;

  void build_csr_from_coo(void *coo_row_ptr,
                          void *coo_col_ptr,
                          void *coo_values_ptr,
                          int nnz) override;

  void nd_spmv(Program *prog, const Ndarray &x, const Ndarray &y);

  void spmv(size_t x, size_t y);

  const void *get_matrix() const override {
    return &matrix_;
  };

  float get_element(int row, int col) const;

  const std::string to_string() const override;

  void *get_row_ptr() const {
    return csr_row_ptr_;
  }
  void *get_col_ind() const {
    return csr_col_ind_;
  }
  void *get_val_ptr() const {
    return csr_val_;
  }
  int get_nnz() const {
    return nnz_;
  }

  void mmwrite(const std::string &filename) override;

 private:
  cusparseSpMatDescr_t matrix_{nullptr};
  void *csr_row_ptr_{nullptr};
  void *csr_col_ind_{nullptr};
  void *csr_val_{nullptr};
  int nnz_{0};
};

std::unique_ptr<SparseMatrix> make_sparse_matrix(
    int rows,
    int cols,
    DataType dt,
    const std::string &storage_format);
std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(int rows,
                                                    int cols,
                                                    DataType dt);
std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(cusparseSpMatDescr_t mat,
                                                    int rows,
                                                    int cols,
                                                    DataType dt,
                                                    void *csr_row_ptr,
                                                    void *csr_col_ind,
                                                    void *csr_val_,
                                                    int nnz);

void make_sparse_matrix_from_ndarray(Program *prog,
                                     SparseMatrix &sm,
                                     const Ndarray &ndarray);
}  // namespace taichi::lang
