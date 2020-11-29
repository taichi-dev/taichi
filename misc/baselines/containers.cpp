#include <atomic>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <map>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

int main() {
  int n = 1024;
  int N = 100000000;
  {
    std::vector<int> a(n);
    auto t = get_time();
    for (int i = 0; i < N; i++) {
      a[(i + 3) & (n - 1)] = i - 3;
    }
    printf("vector write %.4f ns\n", (get_time() - t) / N * 1e9);
  }
  {
    std::unordered_map<int, int> a;
    auto t = get_time();
    for (int i = 0; i < N; i++) {
      a[(i + 3) & (n - 1)] = i - 3;
    }
    printf("unordered_map write %.4f ns\n", (get_time() - t) / N * 1e9);
  }
  {
    std::map<int, int> a;
    auto t = get_time();
    for (int i = 0; i < N; i++) {
      a[(i + 3) & (n - 1)] = i - 3;
    }
    printf("map write %.4f ns\n", (get_time() - t) / N * 1e9);
  }
  return 0;
}

// g++ containers.cpp -o containers -O3 -g -Wall && ./containers
//
// vector write 0.4538 ns
// unordered_map write 6.5586 ns
// map write 35.9977 ns
