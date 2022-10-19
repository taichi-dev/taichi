#pragma once
#include "taichi/program/program.h"

namespace taichi::lang {

inline bool codegen_vector_type(CompileConfig *config) {
  if (config->real_matrix && !config->real_matrix_scalarize) {
    return true;
  }

  return false;
}

}  // namespace taichi::lang
