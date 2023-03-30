#include "sparse_matrix.h"

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#include "Eigen/IterativeLinearSolvers"

namespace taichi::lang {
template <typename EigenT, typename DT>
class CG {
 public:
  CG(SparseMatrix &A, int max_iters, float tol, bool verbose)
      : A_(A), max_iters_(max_iters), tol_(tol), verbose_(verbose) {
    x_ = EigenT::Zero(A_.num_cols());
    b_ = EigenT::Zero(A_.num_rows());
  }

  void set_x(EigenT &x) {
    x_ = x;
  }

  void set_b(EigenT &b) {
    b_ = b;
  }

  void set_x_ndarray(Program *prog, Ndarray &x) {
    size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
    x_ = Eigen::Map<EigenT>((DT *)dX, A_.num_cols());
  }

  void set_b_ndarray(Program *prog, Ndarray &b) {
    size_t db = prog->get_ndarray_data_ptr_as_int(&b);
    b_ = Eigen::Map<EigenT>((DT *)db, A_.num_rows());
  }

  void solve() {
    Eigen::ConjugateGradient<Eigen::SparseMatrix<DT>,
                             Eigen::Lower | Eigen::Upper>
        cg;
    cg.setMaxIterations(max_iters_);
    cg.setTolerance(tol_);
    EigenSparseMatrix<Eigen::SparseMatrix<DT>> &A =
        static_cast<EigenSparseMatrix<Eigen::SparseMatrix<DT>> &>(A_);
    Eigen::SparseMatrix<DT> *A_eigen =
        (Eigen::SparseMatrix<DT> *)A.get_matrix();
    cg.compute(*A_eigen);
    x_ = cg.solve(b_);
    if (verbose_) {
      std::cout << "#iterations:     " << cg.iterations() << std::endl;
      std::cout << "estimated error: " << cg.error() << std::endl;
    }
    is_success_ = !(cg.info());
  }

  EigenT &get_x() {
    return x_;
  }

  bool is_success() {
    return is_success_;
  }

 private:
  SparseMatrix &A_;
  EigenT x_;
  EigenT b_;
  int max_iters_{0};
  DT tol_{0.0f};
  bool verbose_{false};
  bool is_success_{false};
};

template <typename EigenT, typename DT>
std::unique_ptr<CG<EigenT, DT>> make_cg_solver(SparseMatrix &A,
                                               int max_iters,
                                               float tol,
                                               bool verbose) {
  return std::make_unique<CG<EigenT, DT>>(A, max_iters, tol, verbose);
}

class CUCG {
 public:
  CUCG(SparseMatrix &A, int max_iters, float tol, bool verbose)
      : A_(A), max_iters_(max_iters), tol_(tol), verbose_(verbose) {
    init_solver();
  }

  void solve(Program *prog, const Ndarray &x, const Ndarray &b);

 private:
  void init_solver();
  cublasHandle_t handle_;
  SparseMatrix &A_;
  int max_iters_{0};
  float tol_{0.0f};
  bool verbose_{false};
  bool is_success_{false};
};

std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float tol,
                                       bool verbose);
}  // namespace taichi::lang
