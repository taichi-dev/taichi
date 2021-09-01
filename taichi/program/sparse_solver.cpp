#include "sparse_solver.h"

namespace taichi {
namespace lang {

SparseLUSolver::SparseLUSolver() : solver_(std::make_unique<LU>()) {
}

bool SparseLUSolver::compute(const SparseMatrix &sm) {
  solver_->compute(sm.get_matrix());
  if (solver_->info() != Eigen::Success) {
    return false;
  } else
    return true;
}

void SparseLUSolver::analyze_pattern(const SparseMatrix &sm) {
  solver_->analyzePattern(sm.get_matrix());
}

void SparseLUSolver::factorize(const SparseMatrix &sm) {
  solver_->factorize(sm.get_matrix());
}

Eigen::VectorXf SparseLUSolver::solve(
    const Eigen::Ref<const Eigen::VectorXf> &b) {
    return solver_->solve(b);
}

SparseLDLTSolver::SparseLDLTSolver() : solver_(std::make_unique<LDLT>()) {
}

bool SparseLDLTSolver::compute(const SparseMatrix &sm) {
  solver_->compute(sm.get_matrix());
  if (solver_->info() != Eigen::Success) {
    return false;
  } else
    return true;
}

void SparseLDLTSolver::analyze_pattern(const SparseMatrix &sm) {
  solver_->analyzePattern(sm.get_matrix());
}

void SparseLDLTSolver::factorize(const SparseMatrix &sm) {
  solver_->factorize(sm.get_matrix());
}

Eigen::VectorXf SparseLDLTSolver::solve(
    const Eigen::Ref<const Eigen::VectorXf> &b) {
  return solver_->solve(b);
}

SparseLLTSolver::SparseLLTSolver() : solver_(std::make_unique<LLT>()) {
}

bool SparseLLTSolver::compute(const SparseMatrix &sm) {
  solver_->compute(sm.get_matrix());
  if (solver_->info() != Eigen::Success) {
    return false;
  } else
    return true;
}

void SparseLLTSolver::analyze_pattern(const SparseMatrix &sm) {
  solver_->analyzePattern(sm.get_matrix());
}

void SparseLLTSolver::factorize(const SparseMatrix &sm) {
  solver_->factorize(sm.get_matrix());
}

Eigen::VectorXf SparseLLTSolver::solve(
    const Eigen::Ref<const Eigen::VectorXf> &b) {
  return solver_->solve(b);
}

}  // namespace lang
}  // namespace taichi
