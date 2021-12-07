#include "taichi/ir/type_utils.h"

#include "sparse_solver.h"

#include <unordered_map>

#define MAKE_SOLVER(dt, type, order)                                           \
  {                                                                            \
    {#dt, #type, #order}, []() -> std::unique_ptr<SparseSolver> {              \
      using T = Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                                        Eigen::order##Ordering<int>>;          \
      return std::make_unique<EigenSparseSolver<T>>();                         \
    }                                                                          \
  }

using Triplets = std::tuple<std::string, std::string, std::string>;
namespace {
struct key_hash : public std::unary_function<Triplets, std::size_t> {
  std::size_t operator()(const Triplets &k) const {
    auto h1 = std::hash<std::string>{}(std::get<0>(k));
    auto h2 = std::hash<std::string>{}(std::get<1>(k));
    auto h3 = std::hash<std::string>{}(std::get<2>(k));
    return h1 ^ h2 ^ h3;
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

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering) {
  using key_type = Triplets;
  using func_type = std::unique_ptr<SparseSolver> (*)();
  static const std::unordered_map<key_type, func_type, key_hash>
      solver_factory = {
          MAKE_SOLVER(float32, LLT, AMD), MAKE_SOLVER(float32, LLT, COLAMD),
          MAKE_SOLVER(float32, LDLT, AMD), MAKE_SOLVER(float32, LDLT, COLAMD)};
  static const std::unordered_map<std::string, std::string> dt_map = {
      {"f32", "float32"}, {"f64", "float64"}};
  auto it = dt_map.find(taichi::lang::data_type_name(dt));
  if (it == dt_map.end())
    TI_ERROR("Not supported sparse solver data type: {}",
             taichi::lang::data_type_name(dt));

  Triplets solver_key = std::make_tuple(it->second, solver_type, ordering);
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
