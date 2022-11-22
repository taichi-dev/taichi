#pragma once

#include "taichi/ir/type.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/program/program.h"

#include "sparse_matrix.h"

namespace taichi::lang {

class SparseSolver {
 protected:
  int rows_{0};
  int cols_{0};
  DataType dtype_{PrimitiveType::f32};
  bool is_initialized_{false};

 public:
  virtual ~SparseSolver() = default;
  void init_solver(const int rows, const int cols, const DataType dtype) {
    rows_ = rows;
    cols_ = cols;
    dtype_ = dtype;
  }
  virtual bool compute(const SparseMatrix &sm) = 0;
  virtual void analyze_pattern(const SparseMatrix &sm) = 0;
  virtual void factorize(const SparseMatrix &sm) = 0;
  virtual Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b) = 0;
  virtual void solve_rf(Program *prog,
                        const SparseMatrix &sm,
                        const Ndarray &b,
                        Ndarray &x) = 0;
  virtual void solve_cu(Program *prog,
                        const SparseMatrix &sm,
                        const Ndarray &b,
                        Ndarray &x) = 0;
  virtual bool info() = 0;
};

template <class EigenSolver, class EigenMatrix>
class EigenSparseSolver : public SparseSolver {
 private:
  EigenSolver solver_;

 public:
  ~EigenSparseSolver() override = default;
  bool compute(const SparseMatrix &sm) override;
  void analyze_pattern(const SparseMatrix &sm) override;
  void factorize(const SparseMatrix &sm) override;
  Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b) override;
  void solve_cu(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                Ndarray &x) override {
    TI_NOT_IMPLEMENTED;
  };
  void solve_rf(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                Ndarray &x) override;

  bool info() override;
};

class CuSparseSolver : public SparseSolver {
 private:
  csrcholInfo_t info_{nullptr};
  cusolverSpHandle_t cusolver_handle_{nullptr};
  cusparseHandle_t cusparse_handel_{nullptr};
  cusparseMatDescr_t descr_{nullptr};
  void *gpu_buffer_{nullptr};
  bool is_analyzed_{false};
  bool is_factorized_{false};

 public:
  CuSparseSolver();
  ~CuSparseSolver() override = default;
  bool compute(const SparseMatrix &sm) override {
    TI_NOT_IMPLEMENTED;
  };
  void analyze_pattern(const SparseMatrix &sm) override;

  void factorize(const SparseMatrix &sm) override;
  Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b) override {
    TI_NOT_IMPLEMENTED;
  };
  void solve_cu(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                Ndarray &x) override;
  void solve_rf(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                Ndarray &x) override;
  bool info() override {
    TI_NOT_IMPLEMENTED;
  };
};

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering);

std::unique_ptr<SparseSolver> make_cusparse_solver(
    DataType dt,
    const std::string &solver_type,
    const std::string &ordering);
}  // namespace taichi::lang
