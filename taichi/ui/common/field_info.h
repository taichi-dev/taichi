#pragma once
#include "taichi/ui/utils/utils.h"

#include "taichi/ir/type_utils.h"
#include "taichi/ir/snode.h"
#include "taichi/backends/device.h"
#include "taichi/program/program.h"

TI_UI_NAMESPACE_BEGIN

enum class FieldSource : int {
  TaichiCuda = 0,
  TaichiX64 = 1,
  TaichiVulkan = 2,
  TaichiOpenGL = 3
  // support np array / torch tensor in the future?
};

enum class FieldType : int { Scalar = 0, Matrix = 1 };

struct FieldInfo {
  DEFINE_PROPERTY(bool, valid)
  DEFINE_PROPERTY(FieldType, field_type);
  DEFINE_PROPERTY(int, matrix_rows);
  DEFINE_PROPERTY(int, matrix_cols);
  DEFINE_PROPERTY(std::vector<int>, shape);
  DEFINE_PROPERTY(FieldSource, field_source);
  DEFINE_PROPERTY(taichi::lang::DataType, dtype);

  // 'snode' is used by default if a Program is currently present. This
  // is the default behavior and is used automatically when executing
  // Taichi Kernels from Python or with an active Program.
  // 'dev_alloc' is only used when no Program is currently present, for
  // example when loading Taichi AOT modules in an external application
  // and need to provide some information from those kernels to the GUI
  // internal structures.
  using SNodePtr = taichi::lang::SNode *;
  DEFINE_PROPERTY(SNodePtr, snode);
  DEFINE_PROPERTY(taichi::lang::DeviceAllocation, dev_alloc);

  FieldInfo() {
    valid = false;
  }
};

taichi::lang::DevicePtr get_device_ptr(taichi::lang::Program *program,
                                       taichi::lang::SNode *snode);

TI_UI_NAMESPACE_END
