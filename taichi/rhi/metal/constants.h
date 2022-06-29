#pragma once

#include <string>

#include "taichi/inc/constants.h"

namespace taichi {
namespace lang {
namespace metal {

inline constexpr int kMaxNumThreadsGridStrideLoop = 64 * 1024;
inline constexpr int kNumRandSeeds = 64 * 1024;  // 256 KB is nothing
inline constexpr int kMslVersionNone = 0;
inline constexpr int kMaxNumSNodes = taichi_max_num_snodes;

}  // namespace metal
}  // namespace lang
}  // namespace taichi
