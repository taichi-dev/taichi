#pragma once

#include "sparse_matrix.h"

#include "taichi/ir/type.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/program/program.h"

#define DECLARE_EIGEN_LLT_SOLVER(dt, type, order)                    \
  typedef EigenSparseSolver<                                         \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                              Eigen::order##Ordering<int>>,          \
      Eigen::SparseMatrix<dt>>                                       \
      EigenSparseSolver##dt##type##order;

#define DECLARE_EIGEN_LU_SOLVER(dt, type, order)                              \
  typedef EigenSparseSolver<Eigen::Sparse##type<Eigen::SparseMatrix<dt>,      \
                                                Eigen::order##Ordering<int>>, \
                            Eigen::SparseMatrix<dt>>                          \
      EigenSparseSolver##dt##type##order;

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

  template <typename T, typename V>
  void solve_rf(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                const Ndarray &x);
  bool info() override;
};

DECLARE_EIGEN_LLT_SOLVER(float32, LLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LLT, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LDLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LDLT, COLAMD);
DECLARE_EIGEN_LU_SOLVER(float32, LU, AMD);
DECLARE_EIGEN_LU_SOLVER(float32, LU, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LLT, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LDLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LDLT, COLAMD);
DECLARE_EIGEN_LU_SOLVER(float64, LU, AMD);
DECLARE_EIGEN_LU_SOLVER(float64, LU, COLAMD);

class CuSparseSolver : public SparseSolver {
 public:
  enum class SolverType { Cholesky, LU };

 private:
  SolverType solver_type_{SolverType::Cholesky};
  csrcholInfo_t info_{nullptr};
  csrluInfoHost_t lu_info_{nullptr};
  cusolverSpHandle_t cusolver_handle_{nullptr};
  cusparseHandle_t cusparse_handel_{nullptr};
  cusparseMatDescr_t descr_{nullptr};
  void *gpu_buffer_{nullptr};
  void *cpu_buffer_{nullptr};
  bool is_analyzed_{false};
  bool is_factorized_{false};

  int *h_Q_{
      nullptr}; /* <int> n,  B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
  int *d_Q_{nullptr};
  int *h_csrRowPtrB_{nullptr}; /* <int> n+1 */
  int *h_csrColIndB_{nullptr}; /* <int> nnzA */
  float *h_csrValB_{nullptr};  /* <float> nnzA */
  int *h_mapBfromA_{nullptr};  /* <int> nnzA */
  int *d_csrRowPtrB_{nullptr}; /* <int> n+1 */
  int *d_csrColIndB_{nullptr}; /* <int> nnzA */
  float *d_csrValB_{nullptr};  /* <float> nnzA */
 public:
  CuSparseSolver();
  explicit CuSparseSolver(SolverType solver_type) : solver_type_(solver_type) {
    init_solver();
  }
  ~CuSparseSolver() override;
  bool compute(const SparseMatrix &sm) override {
    TI_NOT_IMPLEMENTED;
  };
  void analyze_pattern(const SparseMatrix &sm) override;

  void factorize(const SparseMatrix &sm) override;
  void solve_rf(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                const Ndarray &x);

  bool info() override {
    TI_NOT_IMPLEMENTED;
  };

 private:
  void init_solver();
  void reorder(const CuSparseMatrix &sm);
  void analyze_pattern_cholesky(const SparseMatrix &sm);
  void analyze_pattern_lu(const SparseMatrix &sm);
  void factorize_cholesky(const SparseMatrix &sm);
  void factorize_lu(const SparseMatrix &sm);
  void solve_cholesky(Program *prog,
                      const SparseMatrix &sm,
                      const Ndarray &b,
                      const Ndarray &x);
  void solve_lu(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                const Ndarray &x);
};

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering);

std::unique_ptr<SparseSolver> make_cusparse_solver(
    DataType dt,
    const std::string &solver_type,
    const std::string &ordering);
}  // namespace taichi::lang
