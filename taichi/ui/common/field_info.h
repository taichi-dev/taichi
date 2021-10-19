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

  using SNodePtr = taichi::lang::SNode *;
  DEFINE_PROPERTY(SNodePtr, snode);

  FieldInfo() {
    valid = false;
  }
};

taichi::lang::DevicePtr get_device_ptr(taichi::lang::Program *program,
                                       taichi::lang::SNode *snode);

TI_UI_NAMESPACE_END
