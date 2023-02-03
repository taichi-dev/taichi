#pragma once
#include "taichi/ir/type_utils.h"
#include "taichi/ir/snode.h"
#include "taichi/rhi/device.h"
#include "taichi/program/program.h"

namespace taichi {

namespace ui {

enum class FieldSource : int {
  TaichiNDarray = 0,
  HostMappedPtr = 1,
};

#define DEFINE_PROPERTY(Type, name)                          \
  Type name;                                                 \
  void set_##name(const Type &new_name) { name = new_name; } \
  Type get_##name() { return name; }

struct FieldInfo {
  DEFINE_PROPERTY(bool, valid)
  DEFINE_PROPERTY(std::vector<int>, shape);
  DEFINE_PROPERTY(uint64_t, num_elements);
  DEFINE_PROPERTY(FieldSource, field_source);
  DEFINE_PROPERTY(taichi::lang::DataType, dtype);
  DEFINE_PROPERTY(taichi::lang::DeviceAllocation, dev_alloc);

  FieldInfo() {
    valid = false;
  }
};

taichi::lang::DevicePtr get_device_ptr(taichi::lang::Program *program,
                                       taichi::lang::SNode *snode);

}  // namespace ui

}  // namespace taichi
