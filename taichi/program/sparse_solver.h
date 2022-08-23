#pragma once

#include "taichi/ir/type.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/program/program.h"

#include "sparse_matrix.h"

namespace taichi {
namespace lang {

class SparseSolver {
 public:
  virtual ~SparseSolver() = default;
  virtual bool compute(const SparseMatrix &sm) = 0;
  virtual void analyze_pattern(const SparseMatrix &sm) = 0;
  virtual void factorize(const SparseMatrix &sm) = 0;
  virtual Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b) = 0;
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
                Ndarray &x) override{};
  bool info() override;
};

class CuSparseSolver : public SparseSolver {
 private:
 public:
  CuSparseSolver();
  ~CuSparseSolver() override = default;
  bool compute(const SparseMatrix &sm) override {
    return false;
  };
  void analyze_pattern(const SparseMatrix &sm) override{};
  void factorize(const SparseMatrix &sm) override{};
  Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b) override {
    return b;
  };
  void solve_cu(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                Ndarray &x) override;
  bool info() override {
    return false;
  };
};

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering);

std::unique_ptr<SparseSolver> make_cusparse_solver(
    DataType dt,
    const std::string &solver_type,
    const std::string &ordering);

void cu_solve(Program *prog,
              const Ndarray &row_offsets,
              const Ndarray &col_indices,
              const Ndarray &values,
              int nrows,
              int ncols,
              int nnz,
              const Ndarray &b,
              Ndarray &x);
}  // namespace lang
}  // namespace taichi
