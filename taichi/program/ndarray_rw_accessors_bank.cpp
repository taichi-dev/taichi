#include "taichi/program/ndarray_rw_accessors_bank.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

namespace {
void set_kernel_args(const std::vector<int> &I,
                     int num_active_indices,
                     Kernel::LaunchContextBuilder *launch_ctx) {
  for (int i = 0; i < num_active_indices; i++) {
    launch_ctx->set_arg_int(i, I[i]);
  }
}
void set_kernel_extra_args(const Ndarray *ndarray,
                           int arg_id,
                           Kernel::LaunchContextBuilder *launch_ctx) {
  // accessor kernels are special as they use element_shape as runtime
  // information so it's required to use total_shape here.
  for (int i = 0; i < ndarray->total_shape().size(); ++i) {
    launch_ctx->set_extra_arg_int(arg_id, i, ndarray->total_shape()[i]);
  }
}
}  // namespace

NdarrayRwAccessorsBank::Accessors NdarrayRwAccessorsBank::get(
    Ndarray *ndarray) {
  NdarrayRwKeys keys{ndarray->total_shape().size(), ndarray->dtype};
  if (ndarray_to_kernels_.find(keys) == ndarray_to_kernels_.end()) {
    ndarray_to_kernels_[keys] = {&(program_->get_ndarray_reader(ndarray)),
                                 &(program_->get_ndarray_writer(ndarray))};
  }
  return Accessors(ndarray, ndarray_to_kernels_[keys], program_);
}

NdarrayRwAccessorsBank::Accessors::Accessors(const Ndarray *ndarray,
                                             const RwKernels &kernels,
                                             Program *prog)
    : ndarray_(ndarray),
      prog_(prog),
      reader_(kernels.reader),
      writer_(kernels.writer) {
  TI_ASSERT(reader_ != nullptr);
  TI_ASSERT(writer_ != nullptr);
}

void NdarrayRwAccessorsBank::Accessors::write_float(const std::vector<int> &I,
                                                    float64 val) {
  auto launch_ctx = writer_->make_launch_context();
  set_kernel_args(I, ndarray_->total_shape().size(), &launch_ctx);
  launch_ctx.set_arg_float(ndarray_->total_shape().size(), val);
  launch_ctx.set_arg_external_array(
      ndarray_->total_shape().size() + 1,
      ndarray_->get_device_allocation_ptr_as_int(),
      ndarray_->get_nelement() * ndarray_->get_element_size(),
      /*is_device_allocation=*/true);
  set_kernel_extra_args(ndarray_, ndarray_->total_shape().size() + 1,
                        &launch_ctx);
  prog_->synchronize();
  (*writer_)(launch_ctx);
}

float64 NdarrayRwAccessorsBank::Accessors::read_float(
    const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, ndarray_->total_shape().size(), &launch_ctx);
  launch_ctx.set_arg_external_array(
      ndarray_->total_shape().size(),
      ndarray_->get_device_allocation_ptr_as_int(),
      ndarray_->get_nelement() * ndarray_->get_element_size(),
      /*is_device_allocation=*/true);
  set_kernel_extra_args(ndarray_, ndarray_->total_shape().size(), &launch_ctx);
  (*reader_)(launch_ctx);
  prog_->synchronize();
  auto ret = reader_->get_ret_float(0);
  return ret;
}

// for int32 and int64
void NdarrayRwAccessorsBank::Accessors::write_int(const std::vector<int> &I,
                                                  int64 val) {
  auto launch_ctx = writer_->make_launch_context();
  set_kernel_args(I, ndarray_->total_shape().size(), &launch_ctx);
  launch_ctx.set_arg_int(ndarray_->total_shape().size(), val);
  launch_ctx.set_arg_external_array(
      ndarray_->total_shape().size() + 1,
      ndarray_->get_device_allocation_ptr_as_int(),
      ndarray_->get_nelement() * ndarray_->get_element_size(),
      /*is_device_allocation=*/true);
  set_kernel_extra_args(ndarray_, ndarray_->total_shape().size() + 1,
                        &launch_ctx);
  prog_->synchronize();
  (*writer_)(launch_ctx);
}

int64 NdarrayRwAccessorsBank::Accessors::read_int(const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, ndarray_->total_shape().size(), &launch_ctx);
  launch_ctx.set_arg_external_array(
      ndarray_->total_shape().size(),
      ndarray_->get_device_allocation_ptr_as_int(),
      ndarray_->get_nelement() * ndarray_->get_element_size(),
      /*is_device_allocation=*/true);
  set_kernel_extra_args(ndarray_, ndarray_->total_shape().size(), &launch_ctx);
  (*reader_)(launch_ctx);
  prog_->synchronize();
  auto ret = reader_->get_ret_int(0);
  return ret;
}

uint64 NdarrayRwAccessorsBank::Accessors::read_uint(const std::vector<int> &I) {
  return (uint64)read_int(I);
}

}  // namespace lang
}  // namespace taichi
