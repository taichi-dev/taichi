#pragma once

#include "sparse_matrix.h"

namespace taichi {
namespace lang {

class SparseSolver {
 public:
  virtual ~SparseSolver(){};
  virtual bool compute(const SparseMatrix &sm) = 0;
  virtual void analyze_pattern(const SparseMatrix &sm) = 0;
  virtual void factorize(const SparseMatrix &sm) = 0;
  virtual Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b) = 0;
  virtual bool info() = 0;
};

template <class EigenSolver>
class EigenSparseSolver : public SparseSolver {
 private:
  EigenSolver solver_;

 public:
  virtual ~EigenSparseSolver(){};
  virtual bool compute(const SparseMatrix &sm) override;
  virtual void analyze_pattern(const SparseMatrix &sm) override;
  virtual void factorize(const SparseMatrix &sm) override;
  virtual Eigen::VectorXf solve(
      const Eigen::Ref<const Eigen::VectorXf> &b) override;
  virtual bool info() override;
};

std::unique_ptr<SparseSolver> get_sparse_solver(const std::string &solver_type);

}  // namespace lang
}  // namespace taichi
