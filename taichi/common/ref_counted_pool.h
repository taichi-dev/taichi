#pragma once

#include <map>
#include <mutex>
#include <vector>

#include "taichi/common/core.h"

namespace taichi {

class RefCount {
 public:
  void inc() {
    ref_count++;
  }
  int dec() {
    return --ref_count;
  }
  int count() {
    return ref_count;
  }

 private:
  int ref_count{1};
};

template <class T, bool sync>
class RefCountedPool {
 public:
  void inc(T obj) {
    if constexpr (sync) {
      gc_pool_lock_.lock();
    }

    auto iter = counts_.find(obj);

    if (iter == counts_.end()) {
      counts_[obj] = RefCount();
    } else {
      iter->second.inc();
    }

    if constexpr (sync) {
      gc_pool_lock_.unlock();
    }
  }

  void dec(T obj) {
    if constexpr (sync) {
      gc_pool_lock_.lock();
    }

    auto iter = counts_.find(obj);

    if (iter == counts_.end()) {
      TI_ERROR("Can not find counted reference");
    } else {
      int c = iter->second.dec();
      if (c == 0) {
        gc_pool_.push_back(iter->first);
        counts_.erase(iter);
      }
    }

    if constexpr (sync) {
      gc_pool_lock_.unlock();
    }
  }

  T gc_pop_one(T null) {
    if constexpr (sync) {
      gc_pool_lock_.lock();
    }

    T obj = null;

    if (gc_pool_.size()) {
      obj = gc_pool_.back();
      gc_pool_.pop_back();
    }

    if constexpr (sync) {
      gc_pool_lock_.unlock();
    }

    return obj;
  }

  void gc_remove_all(std::function<void(T)> deallocator) {
    std::lock_guard<std::mutex> lg(gc_pool_lock_);

    for (T obj : gc_pool_) {
      deallocator(obj);
    }
    gc_pool_.clear();
  }

 private:
  std::unordered_map<T, RefCount> counts_;
  std::vector<T> gc_pool_;
  std::mutex gc_pool_lock_;
};

}  // namespace taichi
