#include <numeric>
#include "taichi/program/ndarray.h"

namespace taichi {
namespace lang {

Ndarray::Ndarray(Program *prog,
                 const DataType type,
                 const std::vector<int> &shape)
    : program(prog),
      dtype(type),
      shape(shape),
      data_ptr(nullptr),
      nelement(1),
      element_size(1) {
  LlvmProgramImpl *prog_ = program->get_llvm_program_impl();
  TaichiLLVMContext *tlctx = prog_->get_llvm_context(program->config.arch);

  nelement = std::accumulate(std::begin(shape), std::end(shape), 1,
                             std::multiplies<>());
  element_size = data_type_size(dtype);
  auto *const runtime_jit = tlctx->runtime_jit_module;
  runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", prog_->get_llvm_runtime(),
      nelement * element_size, 1);

  data_ptr = prog_->fetch_result<int *>(taichi_result_buffer_runtime_query_id,
                                        program->result_buffer);
}

void Ndarray::set_item(std::vector<int> &key, int val) {
  int pos = get_linear_index(key);
  data_ptr[pos] = val;
}

int Ndarray::get_item(std::vector<int> &key) const {
  int pos = get_linear_index(key);
  return data_ptr[pos];
}

intptr_t Ndarray::get_data_ptr_as_int() const {
  return reinterpret_cast<std::intptr_t>(data_ptr);
}

int Ndarray::get_element_size() const {
  return element_size;
}

int Ndarray::get_nelement() const {
  return nelement;
}

int Ndarray::get_linear_index(std::vector<int> &key) const {
  assert(key.size() == shape.size());

  // TODO, generalize this to ND?
  int ret{0};
  if (shape.size() == 1) {
    ret = key[0];
  } else if (shape.size() == 2) {
    ret = key[0] * shape[1] + key[1];
  } else {
    assert(0);
  }
  return ret;
}
}  // namespace lang
}  // namespace taichi
