#include "threading.h"

TC_NAMESPACE_BEGIN

void ThreadedTaskManager::run(std::function<void(int)> target, int begin, int end, int num_threads) {
	std::vector<std::thread *> threads;
	std::vector<int> end_points;
	for (int i = 0; i < num_threads; i++) {
		end_points.push_back(i * (end - begin) / num_threads + begin);
	}
	end_points.push_back(end);
	for (int i = 0; i < num_threads; i++) {
		auto func = [&target, i, &end_points]() {
			int begin = end_points[i], end = end_points[i + 1];
			for (int k = begin; k < end; k++) {
				target(k);
			}
		};
		threads.push_back(new std::thread(func));
	}
	for (int i = 0; i < num_threads; i++) {
		threads[i]->join();
		delete threads[i];
	}
}

TC_NAMESPACE_END
