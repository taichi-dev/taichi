#pragma once
#include "common/utils.h"

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

TC_NAMESPACE_BEGIN

class Spinlock {
protected:
	std::atomic<bool> latch;
public:
	Spinlock() {
		latch.store(false);
	}
	void lock() {
		bool unlatched = false;
		while (!latch.compare_exchange_weak(unlatched, true, std::memory_order_acquire)) {
			unlatched = false;
		}
	}
	void unlock() {
		latch.store(false, std::memory_order_release);
	}

	Spinlock(const Spinlock &o) {
		// We just ignore racing condition here...
		latch.store(o.latch.load());
	}

	Spinlock &operator=(const Spinlock &o) {
		// We just ignore racing condition here...
		latch.store(o.latch.load());
	}
};

class ThreadedTaskManager {
public:
	void static run(std::function<void(int)> target, int begin, int end, int num_threads);
};

TC_NAMESPACE_END