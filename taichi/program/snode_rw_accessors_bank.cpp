#include "taichi/program/snode_rw_accessors_bank.h"

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
}  // namespace

SNodeRwAccessorsBank::Accessors SNodeRwAccessorsBank::get(SNode *snode) {
  auto &kernels = snode_to_kernels_[snode];
  if (kernels.reader == nullptr) {
    kernels.reader = &(program_->get_snode_reader(snode));
  }
  if (kernels.writer == nullptr) {
    kernels.writer = &(program_->get_snode_writer(snode));
  }
  return Accessors(snode, kernels, program_);
}

SNodeRwAccessorsBank::Accessors::Accessors(const SNode *snode,
                                           const RwKernels &kernels,
                                           Program *prog)
    : snode_(snode),
      prog_(prog),
      reader_(kernels.reader),
      writer_(kernels.writer) {
  TI_ASSERT(reader_ != nullptr);
  TI_ASSERT(writer_ != nullptr);
}
void SNodeRwAccessorsBank::Accessors::write_float(const std::vector<int> &I,
                                                  float64 val) {
  auto launch_ctx = writer_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  for (int i = 0; i < snode_->num_active_indices; i++) {
    launch_ctx.set_arg_int(i, I[i]);
  }
  launch_ctx.set_arg_float(snode_->num_active_indices, val);
  prog_->synchronize();
  (*writer_)(launch_ctx);
}

float64 SNodeRwAccessorsBank::Accessors::read_float(const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  (*reader_)(launch_ctx);
  prog_->synchronize();
  auto ret = reader_->get_ret_float(0);
  return ret;
}

// for int32 and int64
void SNodeRwAccessorsBank::Accessors::write_int(const std::vector<int> &I,
                                                int64 val) {
  auto launch_ctx = writer_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  launch_ctx.set_arg_int(snode_->num_active_indices, val);
  prog_->synchronize();
  (*writer_)(launch_ctx);
}

int64 SNodeRwAccessorsBank::Accessors::read_int(const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  (*reader_)(launch_ctx);
  prog_->synchronize();
  auto ret = reader_->get_ret_int(0);
  return ret;
}

uint64 SNodeRwAccessorsBank::Accessors::read_uint(const std::vector<int> &I) {
  return (uint64)read_int(I);
}

}  // namespace lang
}  // namespace taichi
