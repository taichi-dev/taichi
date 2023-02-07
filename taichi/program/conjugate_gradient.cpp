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
  // CUBLASDriver::get_instance().cubCreate(&handle);
  // int version;
  // CUBLASDriver::get_instance().cubGetVersion(handle, &version);
  // fmt::print("CUBLAS version: {}\n", version);
#endif
}


void CUCG::solve(Program *prog, const Ndarray &x, const Ndarray &b){
  CuSparseMatrix &A = static_cast<CuSparseMatrix &>(A_);  
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t db = prog->get_ndarray_data_ptr_as_int(&x);
  int m = A.num_rows();

  void *d_Ax = nullptr;
  void *d_r = nullptr;
  void *d_p = nullptr, *d_p_next=nullptr;
  CUDADriver::get_instance().malloc((void**)&d_Ax, sizeof(float) * m);
  CUDADriver::get_instance().malloc((void**)&d_r, sizeof(float) * m);
  CUDADriver::get_instance().malloc((void**)&d_p, sizeof(float) * m);
  CUDADriver::get_instance().malloc((void**)&d_p_next, sizeof(float) * m);

  // // compute d_Ax = A @ x
  A.spmv(dX, size_t(d_Ax));

  // // compute d_r = b - d_Ax
  CUDADriver::get_instance().memcpy_device_to_device((void*)d_r, (void*)db, sizeof(float) * m);
  
  // float alpha = -1.0f;
  // CUBLASDriver::get_instance().cbSaxpy(handle, m, &alpha, (float*)d_Ax, 1, (float*)d_r, 1);

  // float r_norm = 0.0f;
  // CUBLASDriver::get_instance().cbSnrm2(handle, m, (float*)d_r, 1, &r_norm);

  // // if r_norm is sufficiently small, return
  // if (r_norm < 1.0e-6){
  //   return;
  // }

  // // p0 = r0
  // CUDADriver::get_instance().memcpy_device_to_device((void*)d_p, (void*)d_r, sizeof(float) * m);

  // int k=0;
  // while(k < max_iters_){
  //   // sqrt(r^T @ r)
  //   CUBLASDriver::get_instance().cbSnrm2(handle, m, (float*)d_r, 1, &r_norm);
  //   // Ap 
  //   A.spmv(size_t(d_p), size_t(d_Ax));
  //   // p^T @ Ap
  //   float pAp = 0.0f;
  //   CUBLASDriver::get_instance().cbSdot(handle, m, (float*)d_p, 1, (float*)d_Ax, 1, &pAp);
  //   alpha = r_norm * r_norm / pAp;
  //   // x = x + alpha * p
  //   CUBLASDriver::get_instance().cbSaxpy(handle, m, &alpha, (float*)d_p, 1, (float*)dX, 1);
  //   // r = r - alpha * Ap
  //   alpha = -alpha;
  //   CUBLASDriver::get_instance().cbSaxpy(handle, m, &alpha, (float*)d_Ax, 1, (float*)d_r, 1);
  //   // r^T @ r
  //   float r_next_norm = 0.0f;
  //   CUBLASDriver::get_instance().cbSnrm2(handle, m, (float*)d_r, 1, &r_next_norm);
  //   // if r_k+1 is sufficiently small, return
  //   if (r_next_norm < 1.0e-6) return;
  //   float beta = r_next_norm * r_next_norm / r_norm * r_norm;
  //   // p_next = r_next + beta * p
  //   CUDADriver::get_instance().memcpy_device_to_device((void*)d_p_next, (void*)d_r, sizeof(float) * m);
  //   CUBLASDriver::get_instance().cbSaxpy(handle, m, &beta, (float*)d_p, 1, (float*)d_p_next, 1);
  //   CUDADriver::get_instance().memcpy_device_to_device((void*)d_p, (void*)d_p_next, sizeof(float) * m);
  //   k++;
  // }
}


std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float tol,
                                       bool verbose) {
  return std::make_unique<CUCG>(A, max_iters, tol, verbose);
}
} // namespace taichi::lang