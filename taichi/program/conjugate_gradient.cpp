#include "conjugate_gradient.h"
namespace taichi::lang {

void CG::solve() {
  Eigen::ConjugateGradient<Eigen::SparseMatrix<float>,
                           Eigen::Lower | Eigen::Upper>
      cg;
  cg.setMaxIterations(max_iters_);
  cg.setTolerance(tol_);
  EigenSparseMatrix<Eigen::SparseMatrix<float>> &A =
      static_cast<EigenSparseMatrix<Eigen::SparseMatrix<float>> &>(A_);
  Eigen::SparseMatrix<float> *A_eigen =
      (Eigen::SparseMatrix<float> *)A.get_matrix();
  cg.compute(*A_eigen);
  x_ = cg.solve(b_);
  if (verbose_) {
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error() << std::endl;
  }
  is_success_ = !(cg.info());
}

std::unique_ptr<CG> make_cg_solver(SparseMatrix &A,
                                   int max_iters,
                                   float tol,
                                   bool verbose) {
  return std::make_unique<CG>(A, max_iters, tol, verbose);
}

}  // namespace taichi::lang
