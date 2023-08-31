#pragma once

#include "taichi/rhi/device.h"
#include <assert.h>
#include <forward_list>
#include <initializer_list>
#include <unordered_set>
#include <mutex>
#include <type_traits>

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
#define RHI_THROW_UNLESS(cond, exception) \
  if (!(cond))                            \
    throw(exception);

template <typename T>
constexpr auto saturate_uadd(T a, T b) {
  static_assert(std::is_unsigned<T>::value);
  const T c = a + b;
  if (c < a) {
    return std::numeric_limits<T>::max();
  }
  return c;
}

template <typename T>
constexpr auto saturate_usub(T x, T y) {
  static_assert(std::is_unsigned<T>::value);
  T res = x - y;
  res &= -(res <= x);

  return res;
}

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
template <class T>
class SyncedPtrStableObjectList {
  using storage_block = std::array<uint8_t, sizeof(T)>;

 public:
  template <typename... Params>
  T &acquire(Params &&...args) {
    std::lock_guard<std::mutex> _(lock_);

    void *storage = nullptr;
    if (free_nodes_.empty()) {
      storage = objects_.emplace_front().data();
    } else {
      storage = free_nodes_.back();
      free_nodes_.pop_back();
    }
    return *new (storage) T(std::forward<Params>(args)...);
  }

  void release(T *ptr) {
    std::lock_guard<std::mutex> _(lock_);

    ptr->~T();
    free_nodes_.push_back(ptr);
  }

  void clear() {
    std::lock_guard<std::mutex> _(lock_);

    // Transfer to quick look-up
    std::unordered_set<void *> free_nodes_set(free_nodes_.begin(),
                                              free_nodes_.end());
    free_nodes_.clear();
    // Destroy live objects
    for (auto &storage : objects_) {
      T *obj = reinterpret_cast<T *>(storage.data());
      // Call destructor if object is not in the free list (thus live)
      if (free_nodes_set.find(obj) == free_nodes_set.end()) {
        obj->~T();
      }
    }
    // Clear the storage
    objects_.clear();
  }

  ~SyncedPtrStableObjectList() {
    clear();
  }

 private:
  std::mutex lock_;
  std::forward_list<storage_block> objects_;
  std::vector<void *> free_nodes_;
};

// A helper to combine hash
template <class T>
inline void hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// A helper to remove copy constructor
class NonAssignable {
 private:
  NonAssignable(NonAssignable const &);
  NonAssignable &operator=(NonAssignable const &);

 public:
  NonAssignable() {
  }
};

}  // namespace rhi_impl
}  // namespace taichi::lang
