#pragma once

#include "sparse_matrix.h"

#include "taichi/ir/type.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/program/program.h"

#define DECLARE_EIGEN_SOLVER(dt, type, order)                        \
  typedef EigenSparseSolver<                                         \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                              Eigen::order##Ordering<int>>,          \
      Eigen::SparseMatrix<dt>>                                       \
      EigenSparseSolver##dt##type##order;

namespace taichi::lang {

class SparseSolver {
 public:
  virtual ~SparseSolver() = default;
  virtual bool compute(const SparseMatrix &sm) = 0;
  virtual void analyze_pattern(const SparseMatrix &sm) = 0;
  virtual void factorize(const SparseMatrix &sm) = 0;
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
  template <typename T>
  T solve(const T &b);
  bool info() override;
};

DECLARE_EIGEN_SOLVER(float32, LLT, AMD);
DECLARE_EIGEN_SOLVER(float32, LLT, COLAMD);
DECLARE_EIGEN_SOLVER(float32, LDLT, AMD);
DECLARE_EIGEN_SOLVER(float32, LDLT, COLAMD);
DECLARE_EIGEN_SOLVER(float64, LLT, AMD);
DECLARE_EIGEN_SOLVER(float64, LLT, COLAMD);
DECLARE_EIGEN_SOLVER(float64, LDLT, AMD);
DECLARE_EIGEN_SOLVER(float64, LDLT, COLAMD);

typedef EigenSparseSolver<
    Eigen::SparseLU<Eigen::SparseMatrix<float32>, Eigen::AMDOrdering<int>>,
    Eigen::SparseMatrix<float32>>
    EigenSparseSolverfloat32LUAMD;
typedef EigenSparseSolver<
    Eigen::SparseLU<Eigen::SparseMatrix<float32>, Eigen::COLAMDOrdering<int>>,
    Eigen::SparseMatrix<float32>>
    EigenSparseSolverfloat32LUCOLAMD;
typedef EigenSparseSolver<
    Eigen::SparseLU<Eigen::SparseMatrix<float64>, Eigen::AMDOrdering<int>>,
    Eigen::SparseMatrix<float64>>
    EigenSparseSolverfloat64LUAMD;
typedef EigenSparseSolver<
    Eigen::SparseLU<Eigen::SparseMatrix<float64>, Eigen::COLAMDOrdering<int>>,
    Eigen::SparseMatrix<float64>>
    EigenSparseSolverfloat64LUCOLAMD;

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
  void solve_cu(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                Ndarray &x);
  void solve_rf(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                Ndarray &x);
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
