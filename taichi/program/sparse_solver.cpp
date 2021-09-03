#include "sparse_solver.h"

namespace taichi {
namespace lang {

template <class EigenSolver>
bool EigenSparseSolver<EigenSolver>::compute(const SparseMatrix &sm) {
  solver_.compute(sm.get_matrix());
  if (solver_.info() != Eigen::Success) {
    return false;
  } else
    return true;
}
template <class EigenSolver>
void EigenSparseSolver<EigenSolver>::analyze_pattern(const SparseMatrix &sm) {
  solver_.analyzePattern(sm.get_matrix());
}

template <class EigenSolver>
void EigenSparseSolver<EigenSolver>::factorize(const SparseMatrix &sm) {
  solver_.factorize(sm.get_matrix());
}

template <class EigenSolver>
Eigen::VectorXf EigenSparseSolver<EigenSolver>::solve(
    const Eigen::Ref<const Eigen::VectorXf> &b) {
  return solver_.solve(b);
}

template <class EigenSolver>
bool EigenSparseSolver<EigenSolver>::info() {
  return solver_.info() == Eigen::Success;
}

std::unique_ptr<SparseSolver> get_sparse_solver(
    const std::string &solver_type) {
  if (solver_type == "LU") {
    using LU = Eigen::SparseLU<Eigen::SparseMatrix<float32>>;
    return std::make_unique<EigenSparseSolver<LU>>();
  } else if (solver_type == "LDLT") {
    using LDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<float32>>;
    return std::make_unique<EigenSparseSolver<LDLT>>();
  } else if (solver_type == "LLT") {
    using LLT = Eigen::SimplicialLLT<Eigen::SparseMatrix<float32>>;
    return std::make_unique<EigenSparseSolver<LLT>>();
  } else
    TI_ERROR("Not supported sparse solver type: {}", solver_type);
}

}  // namespace lang
}  // namespace taichi
