// #include <mkl.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <cstring>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}
constexpr int n = 1024 * 1024 * 1024 / 4;

int main() {
  auto x = std::vector<float>(n), y = std::vector<float>(n);
  float trash = 0;
  for (int i = 0; i < 10000000; i++) {
    double t = get_time();
    t = get_time();
    std::memset(x.data(), 0, sizeof(float) * n);
    double memset = get_time() - t;
    t = get_time();
    std::memcpy(x.data(), y.data(), sizeof(float) * n);
    double memcpy = get_time() - t;
    printf("CPU time           memset(1) %f ms memcpy(2) %f ms\n",
           memset * 1000, memcpy * 1000);
    printf("Memory bandwidth:  memset(1) %.2f GB/s memcpy(2) %.2f GB/s\n",
           1.0f / memset, 2.0f / memcpy);
  }
  printf("trash %f", trash);
  return 0;
}
