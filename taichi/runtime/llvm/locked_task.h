#pragma once

template <typename T, typename G>
class lock_guard {
 public:
  lock_guard(Ptr lock, const T &func, const G &test) {
#if ARCH_x64 || ARCH_arm64
    mutex_lock_i32(lock);
    if (test())
      func();
    mutex_unlock_i32(lock);
#else
    // CUDA

    auto body = [&]() {
      if (test()) {
        mutex_lock_i32(lock);
        // Memory fences here are necessary since CUDA has a weakly ordered
        // memory model across threads
        grid_memfence();
        if (test())
          func();
        grid_memfence();
        mutex_unlock_i32(lock);
      }
    };

    if (cuda_compute_capability() < 70) {
      // Note that unfortunately critical sections on pre-Pascal (inclusive)
      // devices has undefined behavior (deadlock or not), if more than one
      // threads in a warp try to acquire the same lock.
      // Therefore we need a serialization within a warp
      /*
      bool done = false;
      while (!done) {
        if (atomic_exchange_i32((i32 *)lock, 1) == 1) {
          func();
          done = true;
          mutex_unlock_i32(lock);
        }
      }
      */
      auto fast = false;
      if (fast) {
        auto active_mask = cuda_active_mask();
        auto remaining = active_mask;
        while (remaining) {
          auto leader = cttz_i32(remaining);
          if (warp_idx() == leader) {
            // Memory fences here are necessary since CUDA has a weakly ordered
            // memory model across threads
            body();
          }
          remaining &= ~(1u << leader);
        }
      } else {
        for (int i = 0; i < warp_size(); i++)
          if (warp_idx() == i)
            body();
      }
    } else {
      // post-Volta devices have independent thread scheduling, so mutexes are
      // safe.
      body();
    }
#endif  // CUDA
  }
};

template <typename T, typename G>
void locked_task(void *lock, const T &func, const G &test) {
  lock_guard<T, G> _((Ptr)lock, func, test);
}

template <typename T>
void locked_task(void *lock, const T &func) {
  locked_task(lock, func, []() { return true; });
}
