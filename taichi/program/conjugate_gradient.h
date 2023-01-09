#include "sparse_matrix.h"

#include "Eigen/IterativeLinearSolvers"

namespace taichi::lang {
class CG {
 public:
  CG(SparseMatrix &A, int max_iters, float tol, bool verbose)
      : A_(A), max_iters_(max_iters), tol_(tol), verbose_(verbose) {
    x_ = Eigen::VectorXf::Zero(A_.num_cols());
    b_ = Eigen::VectorXf::Zero(A_.num_rows());
  }

  void set_x(Eigen::VectorXf &x) {
    x_ = x;
  }

  void set_b(Eigen::VectorXf &b) {
    b_ = b;
  }

  void solve();

  Eigen::VectorXf &get_x() {
    return x_;
  }

  bool is_success() {
    return is_success_;
  }

 private:
  SparseMatrix &A_;
  Eigen::VectorXf x_;
  Eigen::VectorXf b_;
  int max_iters_{0};
  float tol_{0.0f};
  bool verbose_{false};
  bool is_success_{false};
};

std::unique_ptr<CG> make_cg_solver(SparseMatrix &A,
                                   int max_iters,
                                   float tol,
                                   bool verbose);

}  // namespace taichi::lang
