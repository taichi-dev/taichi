#include "conjugate_gradient.h"

namespace taichi::lang {
void CUCG::init_solver(){
#if defined(TI_WITH_CUDA)
  if (!CUBLASDriver::get_instance().is_loaded()) {
    bool load_success = CUBLASDriver::get_instance().load_cublas();
    if (!load_success) {
      TI_ERROR("Failed to load cublas library!");
    }
  }
  cublasHandle_t handle;
  CUBLASDriver::get_instance().cbCreate(&handle);
  int version;
  CUBLASDriver::get_instance().cbGetVersion(handle, &version);
  fmt::print("CUBLAS version: {}\n", version);
#endif
}
std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float tol,
                                       bool verbose) {
  return std::make_unique<CUCG>(A, max_iters, tol, verbose);
}
} // namespace taichi::lang