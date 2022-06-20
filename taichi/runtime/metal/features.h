#pragma once

#include <string>

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

inline bool is_supported_sparse_type(SNodeType t) {
  return t == SNodeType::bitmasked || t == SNodeType::dynamic ||
         t == SNodeType::pointer;
}

}  // namespace metal

TLANG_NAMESPACE_END
