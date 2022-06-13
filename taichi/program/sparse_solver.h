#pragma once

#include "taichi/ir/type.h"

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
  bool info() override;
};

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering);

}  // namespace lang
}  // namespace taichi
