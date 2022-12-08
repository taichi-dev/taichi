#pragma once

#include "taichi/rhi/device.h"

namespace taichi::lang {

// Constructs within `rhi_impl` is for implementing RHI
// No public-facing API should use anything within `rhi_impl` namespace
namespace rhi_impl {

template <typename... Ts>
void disabled_function([[maybe_unused]] Ts... C) {
}

#if defined(SPDLOG_H) && defined(TI_WARN)
#define RHI_LOG_ERROR(msg) TI_WARN("RHI Error : {}", msg)
#else
#define RHI_LOG_ERROR(msg) std::cerr << "RHI Error: " << msg << std::endl;
#endif

#define RHI_DEBUG

#ifdef RHI_DEBUG
#if defined(SPDLOG_H) && defined(TI_WARN)
#define RHI_LOG_DEBUG(msg) TI_TRACE("RHI Debug : {}", msg)
#else
#define RHI_LOG_DEBUG(msg) std::cerr << "RHI Debug: " << msg << std::endl;
#endif
#define RHI_DEBUG_SNPRINTF std::snprintf
#else
#define RHI_DEBUG_SNPRINTF taichi::lang::rhi_impl::disabled_function
#define RHI_LOG_DEBUG(msg)
#endif

#define RHI_ASSERT(cond) assert(cond);

// Wrapped return-code & object tuple for simplicity
// Easier to read then std::pair
// NOTE: If an internal function can fail, wrap return object with this!
template <typename T>
struct RhiReturn {
  [[nodiscard]] TiRhiResults result;
  [[nodiscard]] T object;

  RhiReturn(TiRhiResults &result, T &object) : result(result), object(object) {
  }

  RhiReturn(const TiRhiResults &result, const T &object)
      : result(result), object(object) {
  }

  RhiReturn(TiRhiResults &&result, T &&object)
      : result(result), object(std::move(object)) {
  }

  RhiReturn &operator=(const RhiReturn &other) = default;
};

// Bi-directional map, useful for mapping between RHI enums and backend enums
template <typename RhiType, typename BackendType>
struct BidirMap {
  std::unordered_map<RhiType, BackendType> rhi2backend;
  std::unordered_map<BackendType, RhiType> backend2rhi;

  BidirMap(std::initializer_list<std::pair<RhiType, BackendType>> init_list) {
    for (auto &pair : init_list) {
      rhi2backend.insert(pair);
      backend2rhi.insert(std::make_pair(pair.second, pair.first));
    }
  }

  bool exists(RhiType &v) const {
    return rhi2backend.find(v) != rhi2backend.cend();
  }

  const BackendType &at(RhiType &v) const {
    return rhi2backend.at(v);
  }

  bool exists(BackendType &v) const {
    return backend2rhi.find(v) != backend2rhi.cend();
  }

  const RhiType &at(BackendType &v) const {
    return backend2rhi.at(v);
  }
};

}  // namespace rhi_impl
}  // namespace taichi::lang
