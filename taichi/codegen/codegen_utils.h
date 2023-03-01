#pragma once
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

inline bool codegen_vector_type(const CompileConfig &config) {
  return !config.real_matrix_scalarize;
}

inline std::string get_custom_cuda_library_path() {
  std::string path =
      fmt::format("{}/{}", runtime_lib_dir(),
                  "cuda_runtime-cuda-nvptx64-nvidia-cuda-sm_60.bc");
  // check path existance
  if (!path_exists(path)) {
    return "";
  }

  return path;
}

}  // namespace taichi::lang
