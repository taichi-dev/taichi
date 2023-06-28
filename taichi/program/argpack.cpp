#include <numeric>

#include "taichi/program/argpack.h"
#include "taichi/program/program.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi::lang {

ArgPack::ArgPack(Program *prog,
                 const DataType type)
    : prog_(prog) {

  auto* old_type = type->get_type()->as<ArgPackType>();
  auto [argpack_type, alloc_size] = prog->get_argpack_type_with_data_layout(
      old_type, old_type->get_layout());
  dtype = DataType(argpack_type);
  argpack_alloc_ = prog->allocate_memory_on_device(alloc_size,
                                                   prog->result_buffer);
}

ArgPack::~ArgPack() {
  if (prog_) {
    argpack_alloc_.device->dealloc_memory(argpack_alloc_);
  }
}

intptr_t ArgPack::get_device_allocation_ptr_as_int() const {
  // taichi's own argpack's ptr points to its |DeviceAllocation| on the
  // specified device.
  return reinterpret_cast<intptr_t>(&argpack_alloc_);
}

DeviceAllocation ArgPack::get_device_allocation() const {
  return argpack_alloc_;
}

std::size_t ArgPack::get_nelement() const {
  return dtype->as<ArgPackType>()->elements().size();
}

DataType ArgPack::get_data_type() const {
  return dtype;
}

}  // namespace taichi::lang
