#include "sparse_solver.h"

#include <unordered_map>

#define MAKE_SOLVER(type, order)                                              \
  {                                                                           \
    {#type, #order}, []() -> std::unique_ptr<SparseSolver> {                  \
      using T =                                                               \
          Eigen::Simplicial##type<Eigen::SparseMatrix<float32>, Eigen::Lower, \
                                  Eigen::order##Ordering<int>>;               \
      return std::make_unique<EigenSparseSolver<T>>();                        \
    }                                                                         \
  }

namespace {
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};
}  // namespace

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

std::unique_ptr<SparseSolver> make_sparse_solver(const std::string &solver_type,
                                                 const std::string &ordering) {
  using key_type = std::pair<std::string, std::string>;
  using func_type = std::unique_ptr<SparseSolver> (*)();
  static const std::unordered_map<key_type, func_type, pair_hash>
      solver_factory = {
          MAKE_SOLVER(LLT, AMD),
          MAKE_SOLVER(LLT, COLAMD),
          MAKE_SOLVER(LDLT, AMD),
          MAKE_SOLVER(LDLT, COLAMD),
      };
  std::pair<std::string, std::string> solver_key =
      std::make_pair(solver_type, ordering);

  if (solver_factory.find(solver_key) != solver_factory.end()) {
    auto solver_func = solver_factory.at(solver_key);
    return solver_func();
  } else if (solver_type == "LU") {
    using LU = Eigen::SparseLU<Eigen::SparseMatrix<float32>>;
    return std::make_unique<EigenSparseSolver<LU>>();
  } else
    TI_ERROR("Not supported sparse solver type: {}", solver_type);
}

}  // namespace lang
}  // namespace taichi
