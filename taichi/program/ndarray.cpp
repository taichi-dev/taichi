#include <numeric>
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

Ndarray::Ndarray(Program *prog,
                 const DataType type,
                 const std::vector<int> &shape)
    : dtype(type),
      shape(shape),
      num_active_indices(shape.size()),
      data_ptr_(nullptr),
      nelement_(std::accumulate(std::begin(shape),
                                std::end(shape),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype)) {
  TaichiLLVMContext *tlctx =
      prog->get_llvm_program_impl()->get_llvm_context(prog->config.arch);
  auto *const runtime_jit = tlctx->runtime_jit_module;
  TI_TRACE("allocating memory for Ndarray");
  runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned",
      prog->get_llvm_program_impl()->get_llvm_runtime(),
      nelement_ * element_size_, taichi_page_size);
  data_ptr_ = prog->get_llvm_program_impl()->fetch_result<uint64_t *>(
      taichi_result_buffer_runtime_query_id, prog->result_buffer);
}

intptr_t Ndarray::get_data_ptr_as_int() const {
  return reinterpret_cast<intptr_t>(data_ptr_);
}

std::size_t Ndarray::get_element_size() const {
  return element_size_;
}

std::size_t Ndarray::get_nelement() const {
  return nelement_;
}

}  // namespace lang
}  // namespace taichi
