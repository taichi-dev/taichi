#pragma once

#include "taichi/ir/type.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/program/program.h"

#include "sparse_matrix.h"

namespace taichi::lang {

class SparseSolver {
 public:
  virtual ~SparseSolver() = default;
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
                Ndarray &x) override {
    TI_NOT_IMPLEMENTED;
  };

  bool info() override;
};
#define REGISTER_EIGEN_SOLVER(dt, type, order)                       \
  typedef EigenSparseSolver<                                         \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                              Eigen::order##Ordering<int>>,          \
      Eigen::SparseMatrix<dt>>                                       \
      EigenSparseSolver##dt##type##order;

REGISTER_EIGEN_SOLVER(float32, LLT, AMD);
REGISTER_EIGEN_SOLVER(float32, LLT, COLAMD);
REGISTER_EIGEN_SOLVER(float32, LDLT, AMD);
REGISTER_EIGEN_SOLVER(float32, LDLT, COLAMD);

class CuSparseSolver : public SparseSolver {
 private:
  csrcholInfo_t info_{nullptr};
  cusolverSpHandle_t cusolver_handle_{nullptr};
  cusparseHandle_t cusparse_handel_{nullptr};
  cusparseMatDescr_t descr_{nullptr};
  void *gpu_buffer_{nullptr};

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
