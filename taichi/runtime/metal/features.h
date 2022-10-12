#pragma once

#include <string>

#include "taichi/util/lang_util.h"

namespace taichi::lang {
namespace metal {

inline bool is_supported_sparse_type(SNodeType t) {
  return t == SNodeType::bitmasked || t == SNodeType::dynamic ||
         t == SNodeType::pointer;
}

}  // namespace metal

}  // namespace taichi::lang
