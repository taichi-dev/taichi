#pragma once

template <typename T>
class lock_guard {
  Ptr lock;

public:
  lock_guard(Ptr lock, const T &func) : lock(lock) {
#if ARCH_x86_64
    mutex_lock_i32(lock);
    func();
    mutex_unlock_i32(lock);
#else
    // CUDA
    for (int i = 0; i < warp_size(); i++) {
      if (warp_idx() == i) {
        mutex_lock_i32(lock);
        func();
        block_memfence();
        mutex_unlock_i32(lock);
      }
    }
    // Unfortunately critical sections on CUDA has undefined behavior (deadlock
    // or not), if more than one thread in a warp try to acquire locks
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
#endif
  }
};
template <typename T>
void locked_task(void *lock, const T &func) {
  lock_guard<T> _((Ptr)lock, func);
}

