#pragma once
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

inline bool codegen_vector_type(const CompileConfig &config) {
  return !config.real_matrix_scalarize;
}

inline std::string merge_printf_specifier(
    std::optional<std::string> const &from_user,
    std::string const &from_data_type,
    Arch arch) {
  // TODO: implement specifier merging
  return from_data_type;
}

}  // namespace taichi::lang
