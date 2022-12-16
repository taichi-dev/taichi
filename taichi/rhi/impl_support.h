#pragma once

#include "taichi/rhi/device.h"
#include <assert.h>
#include <forward_list>
#include <mutex>

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
#define RHI_USE_TI_LOGGING

#ifdef RHI_DEBUG
#define RHI_DEBUG_SNPRINTF std::snprintf
#ifdef RHI_USE_TI_LOGGING
#include "taichi/common/logging.h"
#define RHI_LOG_DEBUG(msg) TI_TRACE("RHI Debug : {}", msg)
#else
#define RHI_LOG_DEBUG(msg) std::cout << "RHI Debug: " << msg << std::endl;
#endif
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
  [[nodiscard]] RhiResult result;
  [[nodiscard]] T object;

  RhiReturn(RhiResult &result, T &object) : result(result), object(object) {
  }

  RhiReturn(const RhiResult &result, const T &object)
      : result(result), object(object) {
  }

  RhiReturn(RhiResult &&result, T &&object)
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

  BackendType at(RhiType &v) const {
    return rhi2backend.at(v);
  }

  bool exists(BackendType &v) const {
    return backend2rhi.find(v) != backend2rhi.cend();
  }

  RhiType at(BackendType &v) const {
    return backend2rhi.at(v);
  }
};

// A synchronized list of objects that is pointer stable & reuse objects
// It does not mark objects as used, and it does not free objects (destructor is
// not called)
template <class T>
struct SyncedPtrStableObjectList {
  std::mutex lock;
  std::forward_list<T> objects;
  std::vector<T *> free_nodes;

  T &acquire() {
    std::lock_guard<std::mutex> _(lock);
    if (free_nodes.empty()) {
      return objects.emplace_front();
    } else {
      T *obj = free_nodes.back();
      free_nodes.pop_back();
      return *obj;
    }
  }

  void release(T *ptr) {
    std::lock_guard<std::mutex> _(lock);
    free_nodes.push_back(ptr);
  }

  void clear() {
    std::lock_guard<std::mutex> _(lock);
    objects.clear();
    free_nodes.clear();
  }
};

}  // namespace rhi_impl
}  // namespace taichi::lang
