#pragma once
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

inline bool codegen_vector_type(const CompileConfig &config) {
  return !config.real_matrix_scalarize;
}

}  // namespace taichi::lang
